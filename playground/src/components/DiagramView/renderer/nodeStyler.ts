/* ================================================================
   Node Styler — Enterprise visual configuration
   
   Defines glass-morphism node styles with:
   - Type-based color schemes (start, end, process, decision)
   - Depth-layered shadows
   - Glow effects
   - SVG filter definitions
   ================================================================ */

import type { NodeType } from "@/types";

/* ── Node style config ── */
export interface NodeStyle {
  fill: string;
  fillHover: string;
  stroke: string;
  strokeHover: string;
  strokeWidth: number;
  text: string;
  glow: string;
  glowRadius: number;
  shadowOpacity: number;
  radius: number;
  shape: "rect" | "pill" | "diamond";
  dashArray?: string;
}

/* ── Style registry ── */
const STYLES: Record<NodeType, NodeStyle> = {
  start: {
    fill: "rgba(34, 197, 94, 0.08)",
    fillHover: "rgba(34, 197, 94, 0.14)",
    stroke: "rgba(34, 197, 94, 0.45)",
    strokeHover: "rgba(34, 197, 94, 0.7)",
    strokeWidth: 1.5,
    text: "#4ade80",
    glow: "rgba(34, 197, 94, 0.25)",
    glowRadius: 16,
    shadowOpacity: 0.15,
    radius: 28,
    shape: "pill",
  },
  end: {
    fill: "rgba(239, 68, 68, 0.08)",
    fillHover: "rgba(239, 68, 68, 0.14)",
    stroke: "rgba(239, 68, 68, 0.45)",
    strokeHover: "rgba(239, 68, 68, 0.7)",
    strokeWidth: 1.5,
    text: "#f87171",
    glow: "rgba(239, 68, 68, 0.25)",
    glowRadius: 16,
    shadowOpacity: 0.15,
    radius: 28,
    shape: "pill",
  },
  process: {
    fill: "rgba(99, 102, 241, 0.07)",
    fillHover: "rgba(99, 102, 241, 0.13)",
    stroke: "rgba(99, 102, 241, 0.35)",
    strokeHover: "rgba(99, 102, 241, 0.6)",
    strokeWidth: 1.2,
    text: "#a5b4fc",
    glow: "rgba(99, 102, 241, 0.15)",
    glowRadius: 10,
    shadowOpacity: 0.12,
    radius: 12,
    shape: "rect",
  },
  decision: {
    fill: "rgba(245, 158, 11, 0.08)",
    fillHover: "rgba(245, 158, 11, 0.14)",
    stroke: "rgba(245, 158, 11, 0.5)",
    strokeHover: "rgba(245, 158, 11, 0.75)",
    strokeWidth: 1.5,
    text: "#fbbf24",
    glow: "rgba(245, 158, 11, 0.2)",
    glowRadius: 14,
    shadowOpacity: 0.18,
    radius: 0,
    shape: "diamond",
  },
};

/* ── Retry node override (process + purple accent) ── */
export const RETRY_OVERRIDE: Partial<NodeStyle> = {
  fill: "rgba(168, 85, 247, 0.07)",
  fillHover: "rgba(168, 85, 247, 0.13)",
  stroke: "rgba(168, 85, 247, 0.4)",
  strokeHover: "rgba(168, 85, 247, 0.65)",
  text: "#c084fc",
  glow: "rgba(168, 85, 247, 0.2)",
  dashArray: "6 3",
};

/* ── Edge style config ── */
export interface EdgeStyleConfig {
  stroke: string;
  strokeWidth: number;
  opacity: number;
  dashArray?: string;
  markerColor: string;
}

export const EDGE_STYLES = {
  normal: {
    stroke: "rgba(148, 163, 184, 0.3)",
    strokeWidth: 1.4,
    opacity: 1,
    markerColor: "rgba(148, 163, 184, 0.45)",
  } as EdgeStyleConfig,
  retry: {
    stroke: "rgba(168, 85, 247, 0.4)",
    strokeWidth: 1.2,
    opacity: 0.7,
    dashArray: "6 3",
    markerColor: "rgba(168, 85, 247, 0.5)",
  } as EdgeStyleConfig,
  highlight: {
    stroke: "rgba(99, 102, 241, 0.6)",
    strokeWidth: 2,
    opacity: 1,
    markerColor: "rgba(99, 102, 241, 0.7)",
  } as EdgeStyleConfig,
};

/* ── Label style ── */
export interface LabelStyle {
  color: string;
  background: string;
  fontSize: number;
  padding: number;
  borderRadius: number;
}

export const LABEL_STYLE: LabelStyle = {
  color: "rgba(148, 163, 184, 0.8)",
  background: "rgba(11, 15, 23, 0.85)",
  fontSize: 10,
  padding: 4,
  borderRadius: 4,
};

/* ── Legend entries ── */
export interface LegendEntry {
  type: NodeType | "retry";
  label: string;
  color: string;
  shape: "circle" | "diamond" | "rect";
}

export const LEGEND_ENTRIES: LegendEntry[] = [
  { type: "start", label: "Start", color: "#4ade80", shape: "circle" },
  { type: "process", label: "Process", color: "#a5b4fc", shape: "rect" },
  { type: "decision", label: "Decision", color: "#fbbf24", shape: "diamond" },
  { type: "end", label: "End", color: "#f87171", shape: "circle" },
  { type: "retry", label: "Retry", color: "#c084fc", shape: "rect" },
];

/* ================================================================
   PUBLIC API
   ================================================================ */

export function getNodeStyle(type: NodeType, isRetryTarget = false): NodeStyle {
  const base = STYLES[type] ?? STYLES.process;
  if (isRetryTarget) {
    return { ...base, ...RETRY_OVERRIDE };
  }
  return base;
}

/** Shadow filter scaled by depth */
export function getShadowParams(depth: number): {
  offsetY: number;
  blur: number;
  opacity: number;
} {
  const clamped = Math.min(depth, 8);
  return {
    offsetY: 2 + clamped * 0.5,
    blur: 4 + clamped * 2,
    opacity: 0.1 + clamped * 0.02,
  };
}

/** SVG filter definitions (to be injected into <defs>) */
export const SVG_FILTERS = {
  glow: (id: string, color: string, radius: number): string =>
    `<filter id="${id}" x="-50%" y="-50%" width="200%" height="200%">
       <feGaussianBlur in="SourceGraphic" stdDeviation="${radius}" />
       <feFlood flood-color="${color}" />
       <feComposite in2="SourceGraphic" operator="in" />
       <feMerge>
         <feMergeNode />
         <feMergeNode in="SourceGraphic" />
       </feMerge>
     </filter>`,
  shadow: (depth: number): string => {
    const { offsetY, blur, opacity } = getShadowParams(depth);
    return `<feDropShadow dx="0" dy="${offsetY}" stdDeviation="${blur}" flood-color="rgba(0,0,0,${opacity})" />`;
  },
};
