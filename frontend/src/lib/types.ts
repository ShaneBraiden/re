/* The shapes dashboard/app.py returns. Every number here was written by an
 * earlier stage; the frontend renders, it never computes a result. */

/** true: higher is better · false: lower is better · null: closest to 1.0 is best. */
export type Direction = boolean | null;

export type Cell = string | number | boolean | null;
export type Row = Record<string, Cell>;

export interface ConditionParams {
  alpha?: number;
  sigma_r_dn?: number;
  r_psf?: number;
  binning?: number;
  bit_depth?: number;
  order?: string;
  psf_kind?: string;
  severity?: number;
  tier?: string;
}

export interface Condition {
  hash: string;
  label: string;
  severity: number;
  tier?: string;
  params?: ConditionParams;
}

export interface PipelineEntry {
  name: string;
  kind: string;
  available: boolean;
  reason: string;
}

export interface ImageEntry {
  image_id: string;
  reference: string;
}

export interface RunIndex {
  experiment_id: string;
  generated_from: string;
  conditions: Condition[];
  directions: Record<string, Direction>;
  families: Record<string, string>;
  gaps: { pipeline: string; reason: string; image_id: string; degradation: string }[];
  has_values: boolean;
  images: ImageEntry[];
  metrics: string[];
  pipelines: PipelineEntry[];
}

export interface Health {
  ready: boolean;
  upload: boolean;
  experiment?: string;
  default?: string;
  index?: string;
  note?: string;
}

export interface ExperimentSummary {
  id: string;
  default: boolean;
  experiment_id?: string;
  images?: number;
  conditions?: number;
  pipelines?: number;
  available?: number;
  validation?: boolean | null;
  error?: string;
}

export interface ExperimentList {
  default: string | null;
  experiments: ExperimentSummary[];
}

export type MetricValues = Record<string, number | null>;

export interface ValuesResponse {
  image_id: string;
  condition: string;
  values: Record<string, MetricValues>;
}

export interface SeriesResponse {
  image_id: string;
  metric: string;
  series: Record<string, { severity: number; value: number | null; hash: string }[]>;
}

export interface AggregatePoint {
  severity: number;
  hash: string;
  mean: number | null;
  min: number | null;
  max: number | null;
  n: number;
}

export interface AggregateResponse {
  metric: string;
  n_images: number;
  series: Record<string, AggregatePoint[]>;
}

export interface ValidationCheck {
  name: string;
  detail?: string;
  measured?: number;
  synthetic?: number;
  relative_error?: number;
  tolerance?: number;
  passed: boolean;
}

export interface Validation {
  passed: boolean;
  checks?: ValidationCheck[];
  notes?: string[];
}

export interface Manifest {
  experiment_id?: string;
  config_hash?: string;
  seed?: number;
  started_at?: string;
  finished_at?: string;
  git_commit?: string;
  hardware_info?: Record<string, string>;
  packages?: Record<string, string>;
  stages?: Record<string, Record<string, unknown>>;
  weights?: unknown;
}

export interface RunInfo {
  id: string;
  manifest: Manifest | null;
  config: unknown;
  validation: Validation | null;
  gaps: { pipeline: string; reason: string; rows: number }[];
}

export type AnalysisTable =
  | "knees"
  | "significance"
  | "recovery_aggregate"
  | "planned_tests"
  | "main_means"
  | "ablation_means"
  | "nag_training"
  | "noise_validation"
  | "photon_transfer";

export interface DivergenceEntry {
  pipeline?: string;
  diverges: boolean;
  fidelity_metric?: string;
  perceptual_metric?: string;
  n: number;
  note?: string;
  p_value: number | null;
  spearman_rho: number | null;
}

export interface Divergence {
  pooled?: DivergenceEntry;
  per_pipeline?: Record<string, DivergenceEntry>;
}

export interface AnalysisResponse {
  id: string;
  figures_dir: string | null;
  tables: Partial<Record<AnalysisTable, Row[]>>;
  divergence: Divergence | null;
  figures: { name: string; url: string }[];
}

export interface NoiseLevelFunction {
  centres: number[];
  sigma: (number | null)[][];
  estimated?: boolean;
}

export interface Enhancement {
  id: string;
  name?: string;
  kind?: string;
  scale?: number;
  input_shape?: number[];
  output_shape?: number[];
  steps?: { step: string; detail: string }[];
  metrics?: Record<string, MetricValues>;
  seconds?: number;
  features?: Record<string, number>;
  noise_level_function?: NoiseLevelFunction | null;
  checkpoints?: Record<string, string>;
  urls: Record<string, string>;
  modified: number;
}

export interface EnhancementList {
  upload: boolean;
  runs: Enhancement[];
}

export interface EnhanceResult extends Omit<Enhancement, "modified"> {
  outputs: Record<string, string>;
}
