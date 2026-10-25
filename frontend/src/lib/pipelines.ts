/* Pipeline identity. Colour follows the pipeline, fixed here by name - never by
 * its position in whatever subset is switched on, so filtering never repaints
 * the survivors.
 *
 * The bank has more arms than the palette has distinct hues, so identity is
 * composite: an ablation wears its parent's hue with a dash pattern, and the
 * baseline is a neutral reference line rather than a competing series. Legends,
 * tooltips and the table views carry identity too, so it is never colour alone. */

export type Group = "classical" | "learned" | "nag" | "ablation" | "other";

export interface PipelineInfo {
  name: string;
  label: string;
  group: Group;
  color: string;
  dash?: string;
  baseline?: boolean;
  proposed?: boolean;
  blurb?: string;
}

const PIPELINES: Record<string, Omit<PipelineInfo, "name">> = {
  classical_isp: { label: "Classical ISP", group: "classical", color: "--baseline", baseline: true, blurb: "The control: black level, gain, demosaic, white balance, mild denoise and a tone curve. Every arm is measured against it." },
  nlm_denoise: { label: "NLM denoise", group: "classical", color: "--series-2", blurb: "Non-local means on top of the ISP: real denoising with no learned prior." },
  curve_tone: { label: "Curve tone", group: "classical", color: "--series-3", blurb: "Iterative light-enhancement curves with analytic coefficients, in the spirit of Zero-DCE but with no network." },
  rl_deconv: { label: "RL deconvolution", group: "classical", color: "--series-7", blurb: "Denoise, then Richardson–Lucy deconvolution against the recorded PSF." },
  zero_dce: { label: "Zero-DCE", group: "learned", color: "--series-4", blurb: "Zero-reference deep curve estimation (Guo et al., CVPR 2020)." },
  real_esrgan: { label: "Real-ESRGAN", group: "learned", color: "--series-6", blurb: "Blind super-resolution GAN (Wang et al., ICCVW 2021)." },
  nafnet: { label: "NAFNet", group: "learned", color: "--series-6", dash: "7 4" },
  restormer: { label: "Restormer", group: "learned", color: "--series-2", dash: "7 4" },
  learned_isp: { label: "Learned ISP", group: "learned", color: "--series-3", dash: "7 4" },
  zero_dce_ft: { label: "Zero-DCE fine-tuned", group: "learned", color: "--series-4", dash: "7 4", blurb: "Zero-DCE fine-tuned on the testbench corpus, for a fair comparison with NC-DCE." },
  nc_dce: { label: "NC-DCE", group: "nag", color: "--series-8", blurb: "Zero-DCE with the noise map as a fourth input and a loss that stops curves amplifying noise." },
  g_esrgan: { label: "G-ESRGAN", group: "nag", color: "--series-5", blurb: "Real-ESRGAN detail scaled by a Wiener gate, then back-projected onto the measurement." },
  nag: { label: "NAG", group: "nag", color: "--series-1", proposed: true, blurb: "Noise-aware guarded enhancement: NC-DCE → VP-denoise → G-ESRGAN." },
  nag_nodenoise: { label: "NAG − VP-denoise", group: "ablation", color: "--series-1", dash: "7 4" },
  nag_nogate: { label: "NAG − gate", group: "ablation", color: "--series-1", dash: "2 3" },
  nag_nobp: { label: "NAG − back-projection", group: "ablation", color: "--series-1", dash: "9 3 2 3" },
  nc_dce_nosigma: { label: "NC-DCE − σ input", group: "ablation", color: "--series-8", dash: "7 4" },
  nc_dce_nonoise: { label: "NC-DCE − noise loss", group: "ablation", color: "--series-8", dash: "2 3" },
};

export const GROUPS: { id: Group; label: string }[] = [
  { id: "classical", label: "Classical" },
  { id: "learned", label: "Learned" },
  { id: "nag", label: "NAG · proposed" },
  { id: "ablation", label: "Ablations" },
  { id: "other", label: "Other" },
];

export const BASELINE = "classical_isp";

/** The arms the paper's main table compares; the default when a run has too many to read at once. */
export const MAIN_ARMS = ["classical_isp", "nlm_denoise", "curve_tone", "rl_deconv", "zero_dce", "real_esrgan", "nc_dce", "g_esrgan", "nag"];

const ABLATION_SET = ["zero_dce", "zero_dce_ft", "nc_dce", "nc_dce_nosigma", "nc_dce_nonoise", "nag", "nag_nodenoise", "nag_nogate", "nag_nobp"];

const unknown: string[] = [];

export function pipelineInfo(name: string): PipelineInfo {
  const known = PIPELINES[name];
  if (known) return { name, ...known };
  if (!unknown.includes(name)) unknown.push(name);
  const i = unknown.indexOf(name);
  return { name, label: name, group: "other", color: `--series-${(i % 8) + 1}`, dash: "1 3" };
}

export const pipelineLabel = (name: string) => pipelineInfo(name).label;
export const pipelineColor = (name: string) => `var(${pipelineInfo(name).color})`;

export type Preset = "main" | "ablations" | "all";

export const PRESET_LABELS: [Preset, string][] = [
  ["main", "Main arms"],
  ["ablations", "Ablations"],
  ["all", "All"],
];

/** Named subsets of the available arms, or null when the run is small enough to show whole. */
export function armPresets(available: string[]): Record<Preset, string[]> | null {
  const hasNag = available.some((name) => ["nag", "ablation"].includes(pipelineInfo(name).group));
  if (!hasNag) return null;
  const pick = (names: string[]) => names.filter((name) => available.includes(name));
  return { main: pick(MAIN_ARMS), ablations: pick(ABLATION_SET), all: available };
}

/** Put names back in bank order (the server's JSON sorts keys). */
export function inBankOrder(names: string[], bank: string[]): string[] {
  const rank = (name: string) => {
    const i = bank.indexOf(name);
    return i < 0 ? Number.MAX_SAFE_INTEGER : i;
  };
  return [...names].sort((a, b) => rank(a) - rank(b));
}
