/* ================================================================
   Edge Router — Manhattan orthogonal routing with collision avoidance
   
   Produces clean 90° edge paths:
     horizontal → vertical → horizontal
   
   Features:
   - Orthogonal (Manhattan) routing
   - Routing lanes between layers to avoid overlaps
   - Collision detection with offset adjustment
   - Retry edges rendered as arced back-loops
   - Smart label positioning
   
   Performance: O(E) single pass
   ================================================================ */

import type { WorkflowEdge } from "@/types";
import type { LayoutNode } from "./dagLayoutEngine";

/* ── Route segment ── */
export interface RoutePoint {
  x: number;
  y: number;
}

export interface RoutedEdge {
  id: string;
  source: string;
  target: string;
  points: RoutePoint[];
  path: string;          // SVG path d-string
  labelPos: RoutePoint;  // position for label text
  label: string | null;
  isRetry: boolean;
  style: "solid" | "dashed";
  opacity: number;
}

/* ── Port positions (connection points on nodes) ── */
interface Ports {
  top: RoutePoint;
  bottom: RoutePoint;
  left: RoutePoint;
  right: RoutePoint;
}

function getNodePorts(node: LayoutNode): Ports {
  const cx = node.x + node.width / 2;
  return {
    top: { x: cx, y: node.y },
    bottom: { x: cx, y: node.y + node.height },
    left: { x: node.x, y: node.y + node.height / 2 },
    right: { x: node.x + node.width, y: node.y + node.height / 2 },
  };
}

/* ── Determine optimal ports based on relative positions ── */
function selectPorts(
  src: LayoutNode,
  tgt: LayoutNode,
): { srcPort: RoutePoint; tgtPort: RoutePoint; direction: "lr" | "tb" | "rl" } {
  const srcPorts = getNodePorts(src);
  const tgtPorts = getNodePorts(tgt);

  const dx = tgt.x - src.x;
  const dy = tgt.y - src.y;

  // Primary flow: left-to-right (source right → target left)
  if (dx > src.width * 0.5) {
    return {
      srcPort: srcPorts.right,
      tgtPort: tgtPorts.left,
      direction: "lr",
    };
  }

  // Backward edge (target is behind source)
  if (dx < -tgt.width * 0.5) {
    return {
      srcPort: srcPorts.left,
      tgtPort: tgtPorts.right,
      direction: "rl",
    };
  }

  // Vertical flow: same layer — use top/bottom
  if (dy > 0) {
    return {
      srcPort: srcPorts.bottom,
      tgtPort: tgtPorts.top,
      direction: "tb",
    };
  }

  return {
    srcPort: srcPorts.top,
    tgtPort: tgtPorts.bottom,
    direction: "tb",
  };
}

/* ── Manhattan path builder ── */
function buildManhattanPath(
  src: RoutePoint,
  tgt: RoutePoint,
  direction: "lr" | "tb" | "rl",
  laneOffset: number,
): { points: RoutePoint[]; path: string } {
  const points: RoutePoint[] = [src];

  if (direction === "lr") {
    // Horizontal → Vertical → Horizontal
    const midX = (src.x + tgt.x) / 2 + laneOffset;
    points.push({ x: midX, y: src.y });
    points.push({ x: midX, y: tgt.y });
    points.push(tgt);
  } else if (direction === "rl") {
    // Backward: go down, left, down, right
    const offset = 50 + Math.abs(laneOffset);
    const loopY = Math.max(src.y, tgt.y) + offset;
    points.push({ x: src.x, y: loopY });
    points.push({ x: tgt.x, y: loopY });
    points.push(tgt);
  } else {
    // Top-to-bottom: Vertical → Horizontal → Vertical
    const midY = (src.y + tgt.y) / 2 + laneOffset;
    points.push({ x: src.x, y: midY });
    points.push({ x: tgt.x, y: midY });
    points.push(tgt);
  }

  // Build SVG path string with rounded corners
  const path = buildRoundedPath(points, 8);

  return { points, path };
}

/* ── Build SVG path with rounded corners ── */
function buildRoundedPath(points: RoutePoint[], radius: number): string {
  if (points.length < 2) return "";
  if (points.length === 2) {
    return `M${points[0]!.x},${points[0]!.y} L${points[1]!.x},${points[1]!.y}`;
  }

  const parts: string[] = [`M${points[0]!.x},${points[0]!.y}`];

  for (let i = 1; i < points.length - 1; i++) {
    const prev = points[i - 1]!;
    const curr = points[i]!;
    const next = points[i + 1]!;

    // Distance to prev and next
    const dPrev = Math.hypot(curr.x - prev.x, curr.y - prev.y);
    const dNext = Math.hypot(next.x - curr.x, next.y - curr.y);
    const r = Math.min(radius, dPrev / 2, dNext / 2);

    // Points just before and after the corner
    const t1 = r / dPrev;
    const bx = curr.x - (curr.x - prev.x) * t1;
    const by = curr.y - (curr.y - prev.y) * t1;

    const t2 = r / dNext;
    const ax = curr.x + (next.x - curr.x) * t2;
    const ay = curr.y + (next.y - curr.y) * t2;

    parts.push(`L${bx},${by}`);
    parts.push(`Q${curr.x},${curr.y} ${ax},${ay}`);
  }

  const last = points[points.length - 1]!;
  parts.push(`L${last.x},${last.y}`);

  return parts.join(" ");
}

/* ── Retry loop arc builder ── */
function buildRetryPath(
  src: LayoutNode,
  tgt: LayoutNode,
): { points: RoutePoint[]; path: string } {
  const srcPorts = getNodePorts(src);
  const tgtPorts = getNodePorts(tgt);

  // Retry loops go above the nodes
  const topY = Math.min(src.y, tgt.y) - 80;
  const srcP = srcPorts.top;
  const tgtP = tgtPorts.top;

  const points: RoutePoint[] = [srcP, { x: srcP.x, y: topY }, { x: tgtP.x, y: topY }, tgtP];

  // Use a smooth curve for retry
  const cx1 = srcP.x;
  const cy1 = topY - 20;
  const cx2 = tgtP.x;
  const cy2 = topY - 20;
  const path = `M${srcP.x},${srcP.y} C${cx1},${cy1} ${cx2},${cy2} ${tgtP.x},${tgtP.y}`;

  return { points, path };
}

/* ================================================================
   Collision avoidance — assign lane offsets
   ================================================================ */
function assignLaneOffsets(
  edges: WorkflowEdge[],
  nodeMap: Map<string, LayoutNode>,
): Map<string, number> {
  const offsets = new Map<string, number>();

  // Group edges by their routing corridor (same source-layer → target-layer)
  const corridors = new Map<string, string[]>();
  for (const edge of edges) {
    if (edge.style === "retry_loop") continue;
    const src = nodeMap.get(edge.source);
    const tgt = nodeMap.get(edge.target);
    if (!src || !tgt) continue;

    const key = `${src.layer}-${tgt.layer}`;
    if (!corridors.has(key)) corridors.set(key, []);
    corridors.get(key)!.push(edge.id);
  }

  // Assign lane offsets within each corridor
  for (const [, edgeIds] of corridors) {
    if (edgeIds.length <= 1) {
      offsets.set(edgeIds[0]!, 0);
      continue;
    }

    const spacing = 12;
    const half = (edgeIds.length - 1) / 2;
    for (let i = 0; i < edgeIds.length; i++) {
      offsets.set(edgeIds[i]!, (i - half) * spacing);
    }
  }

  return offsets;
}

/* ================================================================
   PUBLIC API — routeEdges
   ================================================================ */
export function routeEdges(
  edges: WorkflowEdge[],
  nodeMap: Map<string, LayoutNode>,
): RoutedEdge[] {
  const laneOffsets = assignLaneOffsets(edges, nodeMap);
  const routed: RoutedEdge[] = [];

  for (const edge of edges) {
    const src = nodeMap.get(edge.source);
    const tgt = nodeMap.get(edge.target);
    if (!src || !tgt) continue;

    const isRetry = edge.style === "retry_loop";
    const label = edge.condition?.label ?? null;

    let points: RoutePoint[];
    let path: string;

    if (isRetry) {
      const result = buildRetryPath(src, tgt);
      points = result.points;
      path = result.path;
    } else {
      const { srcPort, tgtPort, direction } = selectPorts(src, tgt);
      const offset = laneOffsets.get(edge.id) ?? 0;
      const result = buildManhattanPath(srcPort, tgtPort, direction, offset);
      points = result.points;
      path = result.path;
    }

    // Label position: midpoint of the path
    const midIdx = Math.floor(points.length / 2);
    const p1 = points[midIdx - 1] ?? points[0]!;
    const p2 = points[midIdx] ?? points[0]!;
    const labelPos: RoutePoint = {
      x: (p1.x + p2.x) / 2,
      y: (p1.y + p2.y) / 2 - 10,
    };

    routed.push({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      points,
      path,
      labelPos,
      label,
      isRetry,
      style: isRetry ? "dashed" : "solid",
      opacity: isRetry ? 0.5 : 0.7,
    });
  }

  return routed;
}
