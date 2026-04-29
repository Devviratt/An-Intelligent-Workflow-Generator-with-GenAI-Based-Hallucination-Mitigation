/* Canonical TS types - aligned with Python Pydantic models exactly for Render build */

export type GenerationStatus = "idle" | "loading" | "success" | "error";

export type GenerationMode = "workflow" | "flowchart";

export type NodeType = "start" | "end" | "process" | "decision";

export type EdgeStyle = "normal" | "retry_loop";

export interface DomainInfo {
  domain: string;
  display_name: string;
  description: string;
  keywords: string[];
  step_count: number;
  transition_count: number;
}

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
  metadata: Record<string, any>;
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
  metadata: Record<string, any>;
}

export interface GeneratedWorkflow {
  workflow_id: string;
  domain: string;
  title: string;
  description: string;
  is_flowchart: boolean;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  metadata: Record<string, any>;
}

export type IssueSeverity = "error" | "warning" | "info";

export type IssueCategory = "schema" | "logical" | "dependency" | "cycle" | "depth" | "grounding" | "structure" | "transition" | "duplicate" | "retry" | "flowchart" | "reachability";

export interface ValidationIssue {
  severity: IssueSeverity;
  category: IssueCategory;
  message: string;
  node_id: string | null;
  edge_id: string | null;
  details: Record<string, any>;
}

export interface ValidationResult {
  is_valid: boolean;
  issues: ValidationIssue[];
  nodes_validated: number;
  edges_validated: number;
  checks_performed: string[];
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
  observability: any | null;
}

export interface DomainListResponse {
  domains: DomainInfo[];
  count: number;
}

export interface GenerateRequest {
  instruction: string;
  mode: GenerationMode;
  domain_hint?: string;
  include_optional_steps?: boolean;
  custom_steps?: string[];
  use_local_model?: boolean;
  prefer_llm_generation?: boolean;
  minimal?: boolean;
  evaluation_mode?: boolean;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  dataset_count: number;
  modes_supported: string[];
}

export interface ProcessQueryRequest {
  query: string;
  top_k: number;
}

export interface ProcessQueryResponse {
  user_query: string;
  embedding_preview: number[];
  retrieved_chunks: any[];
}

