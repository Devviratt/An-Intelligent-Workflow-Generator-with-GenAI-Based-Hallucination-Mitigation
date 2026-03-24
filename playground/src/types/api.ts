/* ================================================================
   Backend response types — mirrors Python Pydantic models exactly.
   ================================================================ */

// ── Workflow ──

export type NodeType = "start" | "end" | "process" | "decision";

export type EdgeStyle = "normal" | "retry_loop";

export interface Coordinate {
  x: number;
  y: number;
}

export interface LayoutInfo {
  depth: number;
  branch_index: number;
  position: Coordinate;
}

export interface WorkflowNode {
  id: string;
  label: string;
  type: NodeType;
  description: string;
  domain_step_id: string;
  branches: Record<string, string> | null;
  layout: LayoutInfo;
  metadata: Record<string, unknown>;
}

export interface EdgeCondition {
  label: string;
  branch_key: string | null;
}

export interface WorkflowEdge {
  id: string;
  source: string;
  target: string;
  condition: EdgeCondition | null;
  style: EdgeStyle;
  metadata: Record<string, unknown>;
}

export interface GeneratedWorkflow {
  workflow_id: string;
  domain: string;
  title: string;
  description: string;
  is_flowchart: boolean;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  metadata: Record<string, unknown>;
}

// ── Validation ──

export type IssueSeverity = "error" | "warning" | "info";

export type IssueCategory =
  | "schema"
  | "logical"
  | "dependency"
  | "cycle"
  | "depth"
  | "grounding"
  | "structure"
  | "transition"
  | "duplicate"
  | "retry"
  | "flowchart"
  | "reachability";

export interface ValidationIssue {
  severity: IssueSeverity;
  category: IssueCategory;
  message: string;
  node_id: string | null;
  edge_id: string | null;
  details: Record<string, unknown>;
}

export interface ValidationResult {
  is_valid: boolean;
  issues: ValidationIssue[];
  nodes_validated: number;
  edges_validated: number;
  checks_performed: string[];
}

// ── Request / Response ──

export type GenerationMode = "workflow" | "flowchart";

export interface GenerateRequest {
  instruction: string;
  mode: GenerationMode;
  domain_hint?: string;
  include_optional_steps?: boolean;
  custom_steps?: string[];
  evaluation_mode?: boolean;
}

export interface ErrorDetail {
  code: string;
  message: string;
  field: string | null;
}

export interface PipelineMetrics {
  parse_time_ms: number;
  generation_time_ms: number;
  mitigation_time_ms: number;
  validation_time_ms: number;
  layout_time_ms: number;
  total_time_ms: number;
  domain_selected: string;
  domain_confidence: number;
  nodes_generated: number;
  edges_generated: number;
}

export interface GenerateResponse {
  success: boolean;
  workflow: GeneratedWorkflow | null;
  validation: ValidationResult | null;
  metrics: PipelineMetrics;
  errors: ErrorDetail[];
  observability: unknown | null;
}

// ── Domains ──

export interface DomainInfo {
  domain: string;
  display_name: string;
  description: string;
  keywords: string[];
  step_count: number;
  transition_count: number;
}

export interface DomainListResponse {
  domains: DomainInfo[];
  count: number;
}

// ── Health ──

export interface HealthResponse {
  status: string;
  timestamp: string;
  dataset_count: number;
  modes_supported: string[];
}

// ── API Error ──

export interface ApiError {
  error_type: string;
  message: string;
  details: unknown;
}

// ── RAG ──

export interface RetrievedChunk {
  id: string;
  content: string;
  domain: string;
}

export interface ProcessQueryRequest {
  query: string;
  top_k: number;
}

export interface ProcessQueryResponse {
  user_query: string;
  embedding_preview: number[];
  retrieved_chunks: RetrievedChunk[];
}
