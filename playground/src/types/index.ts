/* Core types mirroring Python Pydantic models for frontend compile */

export type GenerationStatus = "idle" | "loading" | "success" | "error";

export type GenerationMode = "workflow" | "flowchart";

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
  type: "start" | "end" | "process" | "decision";
  description: string;
  domain_step_id: string;
  branches?: Record<string, string>;
  layout: LayoutInfo;
  metadata: Record<string, any>;
}

export interface EdgeCondition {
  label: string;
  branch_key?: string;
}

export interface WorkflowEdge {
  id: string;
  source: string;
  target: string;
  condition?: EdgeCondition;
  style: "normal" | "retry_loop";
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

export interface ErrorDetail {
  code: string;
  message: string;
  field?: string;
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
  workflow?: GeneratedWorkflow;
  validation?: any;  // ValidationResult
  metrics: PipelineMetrics;
  errors: ErrorDetail[];
  observability?: any;
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

export interface ValidateRequest {
  workflow: GeneratedWorkflow;
  domain?: string;
}

