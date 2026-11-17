# Text-driven latent flow matching with a VAE

The editable figure is [`text_driven_latent_flow_matching.tex`](text_driven_latent_flow_matching.tex), a standalone TikZ document. The [PDF](text_driven_latent_flow_matching.pdf) is a vector export, and the [PNG](text_driven_latent_flow_matching.png) is a preview.

The figure has two separate pipelines:

- **(a) Training:** a pretrained VAE encoder maps motion $x$ to the target latent $z_1$. Gaussian $z_0$ and $z_1$ define the interpolation $z_t$; a text- and time-conditioned velocity model is fitted to the target $z_1-z_0$.
- **(b) Generation:** Gaussian $z_0$ is integrated under the learned velocity field and text condition $e$ from $t=0$ to $1$. The pretrained VAE decoder maps the resulting $\hat z_1$ to generated motion $\hat x$. Neither $x$, $E_\psi$, nor $z_1$ is required at generation time.

Compile from this directory:

```bash
pdflatex -halt-on-error text_driven_latent_flow_matching.tex
```

To insert the vector PDF into `proposal.tex`, load `graphicx` and use:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figure/text_driven_latent_flow_matching.pdf}
  \caption{Text-conditioned latent flow matching with a pretrained motion VAE. (a) Training encodes a motion $x$ as $z_1$, interpolates it with Gaussian noise $z_0$, and fits the conditional velocity field to $z_1-z_0$. (b) Generation integrates that field from a new $z_0$ under the text condition and decodes the terminal latent into hand motion.}
  \label{fig:latent-flow-pipeline}
\end{figure}
```
