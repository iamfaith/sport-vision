% Rename this file to paper_fall.tex for direct LaTeX compilation.
\documentclass[10pt,conference]{IEEEtran}

\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs,multirow,array}
\usepackage{graphicx}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{siunitx}
\usepackage{url}
\usepackage{xcolor}
\usepackage{cite}

\title{A Rule-Based Temporal Fall Detection Framework from 2D Human Pose Sequences}

\author{%
\IEEEauthorblockN{Author Name(s)}
\IEEEauthorblockA{Affiliation\\
Email: author@domain.com}
}

\begin{document}
\maketitle

\begin{abstract}
We present an interpretable temporal fall-detection framework built on YOLO-based person detection and ONNX pose estimation. The method is designed for online multi-person monitoring with explicit handling of forward falls and partial occlusion. Unlike black-box temporal classifiers, our system computes a biomechanical feature vector from keypoint trajectories and applies score fusion plus rule-based confirmation logic. We provide full formulas for feature extraction, score construction, track association, and event-state transitions, and expose three operating presets (\textit{precision}, \textit{balanced}, and \textit{recall}) for deployment-level precision--recall control.
\end{abstract}

\begin{IEEEkeywords}
fall detection, human pose estimation, temporal modeling, rule-based AI, elderly safety, multi-person tracking
\end{IEEEkeywords}

\section{Introduction}
Fall detection is a safety-critical task in eldercare, smart homes, and surveillance. Deep temporal models can provide strong accuracy but often require large labeled datasets and produce decisions that are difficult to audit. In contrast, an interpretable rule-based temporal pipeline enables direct clinical or engineering validation by exposing each geometric and kinematic factor that contributes to a decision.

This paper formalizes our production-oriented implementation. Contributions are:
\begin{itemize}
    \item A complete keypoint-level mathematical model for fall-related temporal features.
    \item A weighted confidence score with explicit normalization and three rule branches (side-fall, forward-fall, soft forward-fall).
    \item A practical multi-person association strategy (IoU + center-distance fallback) with per-track temporal states.
    \item A preset mechanism (precision/balanced/recall) for direct field tuning.
\end{itemize}

\section{System Overview}
Given input frame $I_t$, the full online pipeline is
\begin{equation}
I_t \rightarrow \text{YOLO detections }\{b_t^j\} \rightarrow \text{pose landmarks }\{\mathbf{k}_t^{j}\} \rightarrow \text{temporal features }\mathbf{f}_t^{j} \rightarrow \text{fall decision}.
\end{equation}

In multi-person mode, each detection is assigned to a persistent track id and each track maintains an independent temporal detector state.

\section{Keypoint-Level Modeling}
\subsection{Landmark Subset and Semantics}
We use a robust subset of landmarks from a MediaPipe-compatible indexing scheme:
\begin{table}[h]
\centering
\caption{Keypoint Index Set Used by the Detector}
\label{tab:keypoints}
\begin{tabular}{ccll}
\toprule
ID & Symbol & Name & Role in Decision \\
\midrule
0  & $N$     & nose           & head-drop cue (forward fall) \\
11 & $S_L$   & left shoulder  & torso orientation/width \\
12 & $S_R$   & right shoulder & torso orientation/width \\
23 & $H_L$   & left hip       & trunk center/compression \\
24 & $H_R$   & right hip      & trunk center/compression \\
25 & $K_L$   & left knee      & knee-collapse + lower fallback \\
26 & $K_R$   & right knee     & knee-collapse + lower fallback \\
27 & $A_L$   & left ankle     & primary lower endpoint \\
28 & $A_R$   & right ankle    & primary lower endpoint \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Coordinate Remapping}
Pose inference runs inside ROI $r_t=(x_0,y_0,w,h)$. ROI coordinates $(x_t^{\text{roi}},y_t^{\text{roi}})$ are remapped to frame coordinates:
\begin{equation}
\label{eq:remap}
x_t = \operatorname{clip}(x_t^{\text{roi}}+x_0,0,W-1),\quad
y_t = \operatorname{clip}(y_t^{\text{roi}}+y_0,0,H-1).
\end{equation}

\subsection{Visibility}
For landmark $i$, visibility is computed as
\begin{equation}
\label{eq:vis}
v_i = \sigma(z_i^{\text{vis}})\,\sigma(z_i^{\text{pres}}),
\end{equation}
where $\sigma(\cdot)$ is sigmoid when logits are provided.

\subsection{Canonical Midpoints and Fallback}
Define
\begin{equation}
S_t=\operatorname{mid}(S_L,S_R),\quad H_t=\operatorname{mid}(H_L,H_R).
\end{equation}
Lower endpoint $L_t$ is selected by availability priority:
\begin{equation}
L_t=
\begin{cases}
\operatorname{mid}(A_L,A_R), & A_L,A_R\text{ valid},\\
\operatorname{mid}(K_L,K_R), & \text{otherwise if }K_L,K_R\text{ valid},\\
\operatorname{mid}(H_L,H_R), & \text{fallback}.
\end{cases}
\end{equation}
A frame is feature-valid only if $\{N_t,S_t,H_t,L_t\}$ exist.

\section{Temporal Feature Definition}
\subsection{Geometry Per Frame}
Given points $\{N_t,S_t,H_t,L_t\}$:
\begin{align}
\text{body\_height}_t &= \max(y)-\min(y), \\
\text{body\_width}_t &= \max\left(d(S_L,S_R),\; d(H_L,H_R)\right), \\
\text{ratio}_t &= \frac{\text{body\_width}_t}{\text{body\_height}_t},
\end{align}
with Euclidean distance
\begin{equation}
 d(P,Q)=\sqrt{(x_P-x_Q)^2+(y_P-y_Q)^2}.
\end{equation}

Torso inclination:
\begin{equation}
\theta_t = \deg\left(\arctan\frac{|x_{S_t}-x_{H_t}|}{|y_{S_t}-y_{H_t}|+\varepsilon}\right).
\end{equation}

Knee angle at $B$ from points $A,B,C$:
\begin{equation}
\angle(A,B,C)=\cos^{-1}\!\left(\frac{(A-B)\cdot(C-B)}{\|A-B\|\|C-B\|+\varepsilon}\right).
\end{equation}
Average knee angle:
\begin{equation}
\kappa_t = \frac{\angle(H_L,K_L,A_L)+\angle(H_R,K_R,A_R)}{2}.
\end{equation}

\subsection{Temporal Differentials}
Using previous valid frame $t-1$ and scale
\begin{equation}
\alpha_t = \max(\text{body\_height}_{t-1},1),
\end{equation}
we define
\begin{align}
\text{vertical\_speed}_t &= \frac{\text{torso\_y}_t-\text{torso\_y}_{t-1}}{\alpha_t}, \\
\text{hip\_drop}_t &= \frac{y_{H_t}-y_{H_{t-1}}}{\alpha_t}, \\
\text{nose\_drop}_t &= \frac{y_{N_t}-y_{N_{t-1}}}{\alpha_t}, \\
\text{shoulder\_drop}_t &= \frac{y_{S_t}-y_{S_{t-1}}}{\alpha_t}, \\
\Delta\theta_t &= \theta_t-\theta_{t-1}, \\
\Delta\rho_t &= \text{ratio}_t-\text{ratio}_{t-1}, \\
\text{knee\_collapse}_t &= \max(\kappa_{t-1}-\kappa_t,0).
\end{align}

\section{Score and Decision Rules}
\subsection{Windowed Statistics}
Let $W$ be the temporal buffer (default 24 valid frames). Define baseline segment $\mathcal{B}$ and recent segment $\mathcal{R}$ (last 5 valid frames). Let
\begin{equation}
\bar{h}_{\mathcal{B}} = \frac{1}{|\mathcal{B}|}\sum_{i\in\mathcal{B}}\text{body\_height}_i,\quad
r_h=\frac{\text{body\_height}_t}{\max(\bar{h}_{\mathcal{B}},1)}.
\end{equation}

\subsection{Normalization}
\begin{equation}
\operatorname{norm}(x;\ell,u)=\operatorname{clip}\left(\frac{x-\ell}{u-\ell},0,1\right).
\end{equation}

\subsection{Confidence Score}
The fused score is
\begin{equation}
\small
\begin{aligned}
\text{score}=
&0.12\min\left(\frac{\text{upright\_count}}{4},1\right)
+0.14\operatorname{norm}(\max\text{vertical\_speed};0.07,0.24) \\
&+0.12\operatorname{norm}(\max\text{hip\_drop};0.08,0.24)
+0.14\operatorname{norm}(\max\text{nose\_drop};0.10,0.30) \\
&+0.10\operatorname{norm}(\max\text{shoulder\_drop};0.08,0.22)
+0.10\operatorname{norm}(\max\text{knee\_collapse};10,35) \\
&+0.10\operatorname{norm}(\max\Delta\theta;14,48)
+0.08\operatorname{norm}(\max\Delta\rho;0.08,0.35) \\
&+0.10\min\left(\frac{\text{lying\_count}}{4},1\right)
+0.08\min\left(\frac{\text{compressed\_count}}{4},1\right)
+0.10\operatorname{norm}(1-r_h;0.07,0.33).
\end{aligned}
\end{equation}

\subsection{Rule Branches}
A fall is confirmed if any branch is true:
\begin{itemize}
    \item \textbf{Side-fall branch}: high torso inclination + high width/height ratio + strong downward dynamics.
    \item \textbf{Forward-fall branch}: strong head/shoulder descent + knee collapse or body compression.
    \item \textbf{Soft-forward branch}: lower-threshold forward pattern for occlusion-heavy scenes.
\end{itemize}

If no branch is confirmed:
\begin{equation}
\text{stage}=
\begin{cases}
\texttt{warning}, & \text{score}\ge s_{\text{warning}},\\
\texttt{monitoring}, & \text{otherwise}.
\end{cases}
\end{equation}

\section{Event State Machine}
\begin{algorithm}[t]
\caption{Online Fall Detection per Track}
\label{alg:fall}
\begin{algorithmic}[1]
\State Initialize buffer, fall\_active $\gets$ False
\For{each valid frame $t$}
    \State Extract features $\mathbf{f}_t$ and update temporal window
    \If{fall\_active}
        \If{recovery condition satisfied}
            \State stage $\gets$ recovered; fall\_active $\gets$ False
        \Else
            \State stage $\gets$ fallen; emit no new event
        \EndIf
    \Else
        \State Compute score and branch confirmations
        \If{any branch true}
            \State stage $\gets$ fallen; emit event; fall\_active $\gets$ True
        \ElsIf{score $\ge s_{\text{warning}}$}
            \State stage $\gets$ warning
        \Else
            \State stage $\gets$ monitoring
        \EndIf
    \EndIf
\EndFor
\end{algorithmic}
\end{algorithm}

Recovery requires both a minimum elapsed duration and majority-upright behavior in the latest short horizon.

\section{Multi-Person Association}
For detection center $c_d$ and track center $c_k$:
\begin{equation}
\delta_c=\frac{\|c_d-c_k\|_2}{\text{frame diagonal}},
\end{equation}
\begin{equation}
\text{match\_score}=0.7\cdot\text{IoU}+0.3\cdot(1-\delta_c).
\end{equation}
A detection is assigned if
\begin{equation}
\text{IoU}\ge \tau_{\text{IoU}}\quad\text{or}\quad \delta_c\le\tau_{\text{center}}.
\end{equation}
Tracks are dropped after $M$ missed frames.

\section{Preset Configuration}
\begin{table*}[t]
\centering
\caption{Preset Parameters (Editable for Deployment)}
\label{tab:presets}
\resizebox{\textwidth}{!}{
\begin{tabular}{lccc}
\toprule
Parameter & Precision & Balanced & Recall \\
\midrule
min\_alert\_frames & 10 & 8 & 6 \\
compressed\_ratio & 0.78 & 0.80 & 0.84 \\
warning\_score & 0.50 & 0.42 & 0.34 \\
side\_score & 0.80 & 0.72 & 0.62 \\
forward\_score & 0.70 & 0.62 & 0.52 \\
soft\_score & 0.58 & 0.50 & 0.42 \\
upright\_angle & 28.0 & 30.0 & 36.0 \\
upright\_ratio & 0.58 & 0.62 & 0.70 \\
lying\_angle\_strong & 58.0 & 55.0 & 48.0 \\
lying\_ratio\_strong & 0.82 & 0.78 & 0.62 \\
lying\_angle\_weak & 52.0 & 48.0 & 42.0 \\
lying\_ratio\_weak & 0.72 & 0.68 & 0.56 \\
\bottomrule
\end{tabular}}
\end{table*}

\section{Symbol Table}
\begin{table}[h]
\centering
\caption{Symbols and Definitions}
\label{tab:symbols}
\begin{tabular}{>{\raggedright\arraybackslash}p{0.18\linewidth}p{0.74\linewidth}}
\toprule
Symbol & Meaning \\
\midrule
$I_t$ & input frame at time $t$ \\
$b_t^j$ & $j$-th person bounding box in frame $t$ \\
$\mathbf{k}_t^j$ & pose keypoint set for person $j$ at frame $t$ \\
$S_t,H_t,L_t,N_t$ & shoulder, hip, lower endpoint, and nose anchors \\
$\theta_t$ & torso inclination angle \\
$\kappa_t$ & averaged knee angle \\
$\bar{h}_{\mathcal{B}}$ & baseline mean body height \\
$r_h$ & height compression ratio \\
$\tau_{\text{IoU}}$ & IoU association threshold \\
$\tau_{\text{center}}$ & center-distance association threshold \\
$M$ & max missed frames before track deletion \\
\bottomrule
\end{tabular}
\end{table}

\section{Experimental Protocol}
\subsection{Dataset Design and Split Strategy}
To make evaluation clinically and engineering-relevant, we recommend two complementary sets:
\begin{itemize}
    \item \textbf{In-domain set} (same camera style as deployment): indoor videos with daily activities, intentional side-falls, and intentional forward-falls.
    \item \textbf{Cross-domain set} (distribution shift): different camera height, lens FoV, illumination, and crowd density.
\end{itemize}

Each video should include event timestamps and person identity for fall intervals. Use event boundaries $[t_s,t_e]$ and optional attributes (fall type, occlusion level, crowd level).

\begin{table}[h]
\centering
\caption{Recommended Dataset Card (Fill Before Running Experiments)}
\label{tab:dataset_card}
\begin{tabular}{lccc}
    oprule
Subset & \#Videos & Duration (h) & \#Fall Events \\
\midrule
Train/Dev & -- & -- & -- \\
Validation & -- & -- & -- \\
Test (in-domain) & -- & -- & -- \\
Test (cross-domain) & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}

    extbf{Split rule:} split by subject and scene (not by frame) to avoid temporal leakage.

\subsection{Event Matching Protocol}
Given predicted fall event time $\hat{t}$ and ground-truth event interval $[t_s,t_e]$, count a true positive if
\begin{equation}
\hat{t} \in [t_s-\Delta,\, t_e+\Delta],
\end{equation}
where $\Delta$ is a tolerance window (e.g., 0.5--1.0 s).

Multiple predictions mapped to the same ground-truth event are counted as one TP plus extra FP.

\subsection{Metrics}
Use event-level metrics as primary endpoints:
\begin{align}
    \text{Precision} &= \frac{TP}{TP+FP}, \\
    \text{Recall} &= \frac{TP}{TP+FN}, \\
    \text{F1} &= \frac{2\cdot\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}.
\end{align}

False alarms per hour:
\begin{equation}
    \text{FAH} = \frac{FP}{\text{total non-fall monitoring hours}}.
\end{equation}

Detection delay (for matched events):
\begin{equation}
    \text{Delay} = \hat{t} - t_s.
\end{equation}

For multi-person consistency, report track stability:
\begin{equation}
    \text{IDSW} = \text{number of track identity switches per hour}.
\end{equation}

\subsection{Baselines}
Evaluate against at least three baselines:
\begin{itemize}
    \item \textbf{B1:} rule-only detector without fused score (single hard branch).
    \item \textbf{B2:} pose features + classical classifier (e.g., SVM/RandomForest) on fixed windows.
    \item \textbf{B3:} optional sequence learner (e.g., LSTM/TCN) for temporal comparison.
\end{itemize}

\subsection{Ablation Matrix}
Run leave-one-component-out ablations:
\begin{itemize}
    \item remove forward-fall branch,
    \item remove knee-collapse term,
    \item remove lower-limb fallback (ankle $\rightarrow$ knee $\rightarrow$ hip),
    \item remove center-distance fallback in track assignment,
    \item replace preset with fixed balanced thresholds only.
\end{itemize}

\subsection{Sensitivity Analysis}
For deployment tuning, sweep the following:
\begin{itemize}
    \item presets: \{precision, balanced, recall\},
    \item association thresholds: $(\tau_{\text{IoU}},\tau_{\text{center}})$,
    \item max missed frames $M$,
    \item frame skip factor.
\end{itemize}

\subsection{Implementation and Reproducibility}
Use fixed random seeds where stochastic components are introduced, and report hardware details (CPU/GPU model, RAM, OS, Python version).

Practical command template (single video):
\begin{verbatim}
python infer_video_fall_onnx.py <video>.mp4 \
  --fall-preset <precision|balanced|recall> \
  --multi-person --max-persons 5 \
  --track-iou-threshold 0.22 \
  --track-center-threshold 0.18 \
  --track-max-missed 24 \
  --output-json <out>.json -o <out>.mp4
\end{verbatim}

Batch-evaluation idea:
\begin{enumerate}
    \item Run all videos with each preset and store JSON outputs.
    \item Convert event JSON to TP/FP/FN by matching against annotation files.
    \item Aggregate by scene type (single person, crowded, occluded, forward-fall heavy).
\end{enumerate}

\section{Results and Reporting Tables}
\begin{table}[h]
\centering
\caption{Main Comparison (Event-Level)}
\label{tab:main_results}
\begin{tabular}{lcccc}
\toprule
Method & Precision & Recall & F1 & FAH \\
\midrule
Baseline-1 & -- & -- & -- & -- \\
Baseline-2 & -- & -- & -- & -- \\
Ours (precision) & -- & -- & -- & -- \\
Ours (balanced) & -- & -- & -- & -- \\
Ours (recall) & -- & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{Preset Sensitivity (Recommended)}
\label{tab:preset_sensitivity}
\begin{tabular}{lccccc}
    oprule
Preset & Precision & Recall & F1 & FAH & Delay (s) \\
\midrule
precision & -- & -- & -- & -- & -- \\
balanced & -- & -- & -- & -- & -- \\
recall & -- & -- & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{Ablation Study}
\label{tab:ablation}
\begin{tabular}{lcccc}
\toprule
Variant & Prec. & Rec. & F1 & Delay \\
\midrule
Full model & -- & -- & -- & -- \\
- forward branch & -- & -- & -- & -- \\
- knee collapse term & -- & -- & -- & -- \\
- fallback logic & -- & -- & -- & -- \\
- center-distance association & -- & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Qualitative Analysis}
\begin{figure}[h]
\centering
\fbox{\parbox[c][3.2cm][c]{0.9\linewidth}{\centering Placeholder for qualitative failure/success cases}}
\caption{Qualitative examples: side-fall, forward-fall, occlusion, and recovery.}
\label{fig:qualitative}
\end{figure}

\section{Discussion}
Key points to discuss after filling tables:
\begin{itemize}
    \item Precision--recall movement across the three presets.
    \item Whether forward-fall recall gains come with acceptable FAH increase.
    \item Impact of track association on event continuity in crowded scenes.
    \item Failure modes: extreme top-view, severe blur, prolonged full occlusion.
\end{itemize}

\section{Conclusion}
We provide a deployable and interpretable temporal fall detector with explicit keypoint-level formulas, multi-person tracking integration, and configurable precision--recall presets suitable for real-world monitoring.

\section*{Ethical Considerations}
[Describe privacy safeguards, data minimization, and human-in-the-loop alert verification requirements.]

\section*{Acknowledgment}
[Funding and contributor acknowledgments, if any.]

\bibliographystyle{IEEEtran}
\begin{thebibliography}{00}
\bibitem{placeholder1} Author, ``Title,'' Journal/Conference, Year.
\bibitem{placeholder2} Author, ``Title,'' Journal/Conference, Year.
\end{thebibliography}

\end{document}
