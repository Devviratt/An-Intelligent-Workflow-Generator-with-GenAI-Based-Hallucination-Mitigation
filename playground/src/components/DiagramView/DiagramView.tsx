/* ================================================================
   DiagramView — Enterprise Diagram Engine
   
   Composes:
   - dagLayoutEngine → hierarchical Sugiyama layout
   - crossingMinimizer → barycenter edge crossing reduction
   - edgeRouter → Manhattan orthogonal routing
   - nodeStyler → glass-morphism visual hierarchy
   - interactionEngine → zero-rerender CSS-transform zoom/pan
   
   Features:
   - Animated node entrance (fade + rise)
   - Animated edge drawing
   - Hover highlight & selected focus ring
   - Mini legend panel
   - Fit-to-view on data change
   - Fullscreen toggle
   ================================================================ */

import {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { GeneratedWorkflow, WorkflowEdge } from "@/types";
import {
  computeLayout,
  routeEdges,
  getNodeStyle,
  getShadowParams,
  EDGE_STYLES,
  LABEL_STYLE,
  LEGEND_ENTRIES,
  computeFitToView,
  applyTransform,
  zoomAtPoint,
  zoomByStep,
  panBy,
  createDragState,
  startDrag,
  updateDrag,
  endDrag,
} from "./renderer";
import type {
  LayoutNode,
  LayoutResult,
  RoutedEdge,
  ViewportState,
} from "./renderer";
import styles from "./DiagramView.module.css";

/* ── Constants ── */
const ZOOM_STEP = 0.15;

/* ================================================================
   SVG Sub-components
   ================================================================ */

/* ── Node renderer ── */
const SvgNode = memo(function SvgNode({
  node,
  index,
  isRetryTarget,
  isHovered,
  isSelected,
  onHover,
  onClick,
}: {
  node: LayoutNode;
  index: number;
  isRetryTarget: boolean;
  isHovered: boolean;
  isSelected: boolean;
  onHover: (id: string | null) => void;
  onClick: (id: string) => void;
}) {
  const style = getNodeStyle(node.type, isRetryTarget);
  const shadow = getShadowParams(node.depth);
  const { x, y, width, height } = node;
  const cx = x + width / 2;
  const cy = y + height / 2;

  const fill = isHovered ? style.fillHover : style.fill;
  const stroke = isHovered || isSelected ? style.strokeHover : style.stroke;
  const strokeW = isSelected ? style.strokeWidth + 0.6 : style.strokeWidth;

  // Truncate label
  const maxLen = Math.floor(width / 8);
  const label =
    node.label.length > maxLen
      ? node.label.slice(0, maxLen - 1) + "…"
      : node.label;

  // Animation delay based on order
  const delay = index * 40;

  const common = {
    className: styles.nodeShape,
    style: {
      animationDelay: `${delay}ms`,
      filter: `drop-shadow(0 ${shadow.offsetY}px ${shadow.blur}px rgba(0,0,0,${shadow.opacity}))`,
    },
    onMouseEnter: () => onHover(node.id),
    onMouseLeave: () => onHover(null),
    onClick: () => onClick(node.id),
  };

  if (style.shape === "diamond") {
    // Diamond for decision nodes
    const hw = width / 2;
    const hh = height / 2 + 6;
    const points = `${cx},${cy - hh} ${cx + hw},${cy} ${cx},${cy + hh} ${cx - hw},${cy}`;

    return (
      <g {...common}>
        {/* Glow layer */}
        <polygon
          points={points}
          fill={style.glow}
          stroke="none"
          className={styles.nodeGlow}
        />
        {/* Main shape */}
        <polygon
          points={points}
          fill={fill}
          stroke={stroke}
          strokeWidth={strokeW}
        />
        {/* Focus ring */}
        {isSelected && (
          <polygon
            points={points}
            fill="none"
            stroke={style.strokeHover}
            strokeWidth={0.5}
            strokeDasharray="4 2"
            className={styles.focusRing}
            transform={`translate(${cx}, ${cy}) scale(1.08) translate(${-cx}, ${-cy})`}
          />
        )}
        {/* Label */}
        <text
          x={cx}
          y={cy + 1}
          textAnchor="middle"
          dominantBaseline="central"
          fill={style.text}
          fontSize={11}
          fontWeight={600}
          fontFamily="var(--font-sans)"
          className={styles.nodeLabel}
        >
          {label}
        </text>
      </g>
    );
  }

  // Pill or rect
  const rx = style.shape === "pill" ? height / 2 : style.radius;

  return (
    <g {...common}>
      {/* Glow layer */}
      <rect
        x={x - 3}
        y={y - 3}
        width={width + 6}
        height={height + 6}
        rx={rx + 2}
        fill={style.glow}
        stroke="none"
        className={styles.nodeGlow}
      />
      {/* Main shape */}
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx={rx}
        fill={fill}
        stroke={stroke}
        strokeWidth={strokeW}
        strokeDasharray={style.dashArray}
      />
      {/* Focus ring */}
      {isSelected && (
        <rect
          x={x - 4}
          y={y - 4}
          width={width + 8}
          height={height + 8}
          rx={rx + 3}
          fill="none"
          stroke={style.strokeHover}
          strokeWidth={0.5}
          strokeDasharray="4 2"
          className={styles.focusRing}
        />
      )}
      {/* Label */}
      <text
        x={cx}
        y={cy + 1}
        textAnchor="middle"
        dominantBaseline="central"
        fill={style.text}
        fontSize={12}
        fontWeight={600}
        fontFamily="var(--font-sans)"
        className={styles.nodeLabel}
      >
        {label}
      </text>
    </g>
  );
});

/* ── Edge renderer ── */
const SvgEdge = memo(function SvgEdge({
  edge,
  index,
  isHighlighted,
}: {
  edge: RoutedEdge;
  index: number;
  isHighlighted: boolean;
}) {
  const esConfig = edge.isRetry ? EDGE_STYLES.retry : EDGE_STYLES.normal;
  const activeConfig = isHighlighted ? EDGE_STYLES.highlight : esConfig;
  const markerId = edge.isRetry ? "arrowRetry" : "arrowNormal";
  const delay = index * 20;

  return (
    <g
      className={styles.edgePath}
      style={{ animationDelay: `${delay}ms` }}
    >
      <path
        d={edge.path}
        fill="none"
        stroke={activeConfig.stroke}
        strokeWidth={activeConfig.strokeWidth}
        strokeDasharray={activeConfig.dashArray}
        opacity={activeConfig.opacity}
        markerEnd={`url(#${markerId})`}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {edge.label && (
        <g>
          <rect
            x={edge.labelPos.x - edge.label.length * 3 - LABEL_STYLE.padding}
            y={edge.labelPos.y - 7}
            width={edge.label.length * 6 + LABEL_STYLE.padding * 2}
            height={14}
            rx={LABEL_STYLE.borderRadius}
            fill={LABEL_STYLE.background}
          />
          <text
            x={edge.labelPos.x}
            y={edge.labelPos.y + 3}
            textAnchor="middle"
            fill={LABEL_STYLE.color}
            fontSize={LABEL_STYLE.fontSize}
            fontFamily="var(--font-sans)"
            fontWeight={500}
          >
            {edge.label}
          </text>
        </g>
      )}
    </g>
  );
});

/* ── Legend panel ── */
function Legend() {
  return (
    <div className={styles.legend}>
      {LEGEND_ENTRIES.map((entry) => (
        <div key={entry.type} className={styles.legendItem}>
          <span
            className={styles.legendIcon}
            data-shape={entry.shape}
            style={{ borderColor: entry.color, color: entry.color }}
          />
          <span className={styles.legendLabel}>{entry.label}</span>
        </div>
      ))}
    </div>
  );
}

/* ================================================================
   Main DiagramView Component
   ================================================================ */

interface DiagramViewProps {
  workflow: GeneratedWorkflow | null;
}

export const DiagramView = memo(function DiagramView({
  workflow,
}: DiagramViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgGroupRef = useRef<SVGGElement>(null);
  const viewportRef = useRef<ViewportState>({ x: 0, y: 0, scale: 1 });
  const dragRef = useRef(createDragState());
  const rafRef = useRef(0);
  const [, forceUpdate] = useState(0); // for toolbar zoom label only
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  /* ── Compute layout + routes ── */
  const layoutResult: LayoutResult | null = useMemo(() => {
    if (!workflow || workflow.nodes.length === 0) return null;
    return computeLayout(workflow);
  }, [workflow]);

  const routedEdges: RoutedEdge[] = useMemo(() => {
    if (!layoutResult || !workflow) return [];
    return routeEdges(workflow.edges, layoutResult.nodes);
  }, [layoutResult, workflow]);

  // Identify retry targets for styling
  const retryTargets = useMemo(() => {
    if (!workflow) return new Set<string>();
    return new Set(
      workflow.edges
        .filter((e: WorkflowEdge) => e.style === "retry_loop")
        .map((e: WorkflowEdge) => e.target),
    );
  }, [workflow]);

  // Sorted nodes for rendering order (by layer then order)
  const sortedNodes = useMemo(() => {
    if (!layoutResult) return [];
    return Array.from(layoutResult.nodes.values()).sort(
      (a, b) => a.layer - b.layer || a.order - b.order,
    );
  }, [layoutResult]);

  // Connected edges for hover highlighting
  const connectedEdges = useMemo(() => {
    if (!hoveredNode || !workflow) return new Set<string>();
    return new Set(
      workflow.edges
        .filter((e) => e.source === hoveredNode || e.target === hoveredNode)
        .map((e) => e.id),
    );
  }, [hoveredNode, workflow]);

  /* ── Viewport helpers ── */
  const syncTransform = useCallback(() => {
    if (svgGroupRef.current) {
      applyTransform(svgGroupRef.current, viewportRef.current);
    }
  }, []);

  const triggerZoomLabel = useCallback(() => {
    forceUpdate((n) => n + 1);
  }, []);

  /* ── Fit to view on data change ── */
  useEffect(() => {
    if (!layoutResult || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    viewportRef.current = computeFitToView(
      layoutResult.canvasWidth,
      layoutResult.canvasHeight,
      rect.width,
      rect.height,
    );
    syncTransform();
    triggerZoomLabel();
  }, [layoutResult, syncTransform, triggerZoomLabel]);

  /* ── Mouse wheel zoom ── */
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const handler = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      viewportRef.current = zoomAtPoint(
        viewportRef.current,
        e.deltaY,
        mx,
        my,
      );
      syncTransform();
      triggerZoomLabel();
    };

    el.addEventListener("wheel", handler, { passive: false });
    return () => el.removeEventListener("wheel", handler);
  }, [syncTransform, triggerZoomLabel]);

  /* ── Mouse drag pan ── */
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return;
      startDrag(dragRef.current, viewportRef.current, e.clientX, e.clientY);
    },
    [],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const result = updateDrag(dragRef.current, e.clientX, e.clientY);
      if (!result) return;
      cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(() => {
        viewportRef.current = {
          x: result.x,
          y: result.y,
          scale: viewportRef.current.scale,
        };
        syncTransform();
      });
    },
    [syncTransform],
  );

  const handleMouseUp = useCallback(() => {
    endDrag(dragRef.current);
  }, []);

  /* ── Toolbar actions ── */
  const handleZoomIn = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    viewportRef.current = zoomByStep(
      viewportRef.current,
      ZOOM_STEP,
      rect.width / 2,
      rect.height / 2,
    );
    syncTransform();
    triggerZoomLabel();
  }, [syncTransform, triggerZoomLabel]);

  const handleZoomOut = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    viewportRef.current = zoomByStep(
      viewportRef.current,
      -ZOOM_STEP,
      rect.width / 2,
      rect.height / 2,
    );
    syncTransform();
    triggerZoomLabel();
  }, [syncTransform, triggerZoomLabel]);

  const handleFitToView = useCallback(() => {
    if (!layoutResult || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    viewportRef.current = computeFitToView(
      layoutResult.canvasWidth,
      layoutResult.canvasHeight,
      rect.width,
      rect.height,
    );
    syncTransform();
    triggerZoomLabel();
  }, [layoutResult, syncTransform, triggerZoomLabel]);

  const handleToggleFullscreen = useCallback(() => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().catch(() => {});
      setIsFullscreen(true);
    } else {
      document.exitFullscreen().catch(() => {});
      setIsFullscreen(false);
    }
  }, []);

  /* ── Keyboard shortcuts ── */
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "=" || e.key === "+") handleZoomIn();
      else if (e.key === "-") handleZoomOut();
      else if (e.key === "0") handleFitToView();
      else if (e.key === "Escape") setSelectedNode(null);
      else if (e.key === "ArrowUp") {
        viewportRef.current = panBy(viewportRef.current, 0, 40);
        syncTransform();
      } else if (e.key === "ArrowDown") {
        viewportRef.current = panBy(viewportRef.current, 0, -40);
        syncTransform();
      } else if (e.key === "ArrowLeft") {
        viewportRef.current = panBy(viewportRef.current, 40, 0);
        syncTransform();
      } else if (e.key === "ArrowRight") {
        viewportRef.current = panBy(viewportRef.current, -40, 0);
        syncTransform();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [handleZoomIn, handleZoomOut, handleFitToView, syncTransform]);

  /* ── Node interaction ── */
  const handleNodeHover = useCallback((id: string | null) => {
    setHoveredNode(id);
  }, []);

  const handleNodeClick = useCallback((id: string) => {
    setSelectedNode((prev) => (prev === id ? null : id));
  }, []);

  /* ── Empty state ── */
  if (!workflow || !layoutResult) {
    return (
      <div className={styles.container}>
        <div className={styles.emptyState}>
          <div className={styles.emptyIcon}>◇</div>
          <p className={styles.emptyText}>
            Generate a workflow to see the diagram
          </p>
        </div>
      </div>
    );
  }

  const zoomPercent = Math.round(viewportRef.current.scale * 100);

  return (
    <div
      ref={containerRef}
      className={styles.container}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      tabIndex={0}
    >
      {/* Grid background */}
      <div className={styles.gridBg} />

      {/* Toolbar */}
      <div className={styles.toolbar}>
        <button
          className={styles.toolBtn}
          onClick={handleZoomIn}
          title="Zoom in (+)"
          type="button"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M7 2v10M2 7h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
        </button>
        <span className={styles.zoomLabel}>{zoomPercent}%</span>
        <button
          className={styles.toolBtn}
          onClick={handleZoomOut}
          title="Zoom out (−)"
          type="button"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M2 7h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
        </button>
        <div className={styles.toolDivider} />
        <button
          className={styles.toolBtn}
          onClick={handleFitToView}
          title="Fit to view (0)"
          type="button"
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M1 5V1h4M9 1h4v4M13 9v4H9M5 13H1V9" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
        <button
          className={styles.toolBtn}
          onClick={handleToggleFullscreen}
          title="Fullscreen"
          type="button"
        >
          {isFullscreen ? (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M5 1v4H1M9 1v4h4M5 13V9H1M9 13V9h4" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M1 5V1h4M9 1h4v4M13 9v4H9M5 13H1V9" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          )}
        </button>
      </div>

      {/* SVG Canvas */}
      <svg
        width="100%"
        height="100%"
        className={styles.svg}
        style={{ cursor: dragRef.current.active ? "grabbing" : "grab" }}
      >
        <defs>
          {/* Normal arrow marker */}
          <marker
            id="arrowNormal"
            markerWidth="10"
            markerHeight="8"
            refX="10"
            refY="4"
            orient="auto"
            markerUnits="userSpaceOnUse"
          >
            <path
              d="M0,0 L10,4 L0,8 L2,4 Z"
              fill={EDGE_STYLES.normal.markerColor}
            />
          </marker>
          {/* Retry arrow marker */}
          <marker
            id="arrowRetry"
            markerWidth="10"
            markerHeight="8"
            refX="10"
            refY="4"
            orient="auto"
            markerUnits="userSpaceOnUse"
          >
            <path
              d="M0,0 L10,4 L0,8 L2,4 Z"
              fill={EDGE_STYLES.retry.markerColor}
            />
          </marker>
        </defs>

        <g ref={svgGroupRef}>
          {/* Edges (behind nodes) */}
          {routedEdges.map((edge, i) => (
            <SvgEdge
              key={edge.id}
              edge={edge}
              index={i}
              isHighlighted={connectedEdges.has(edge.id)}
            />
          ))}
          {/* Nodes */}
          {sortedNodes.map((node, i) => (
            <SvgNode
              key={node.id}
              node={node}
              index={i}
              isRetryTarget={retryTargets.has(node.id)}
              isHovered={hoveredNode === node.id}
              isSelected={selectedNode === node.id}
              onHover={handleNodeHover}
              onClick={handleNodeClick}
            />
          ))}
        </g>
      </svg>

      {/* Stats bar */}
      <div className={styles.statsBar}>
        <span>
          {sortedNodes.length} node{sortedNodes.length !== 1 ? "s" : ""}
        </span>
        <span className={styles.statsDot}>·</span>
        <span>
          {routedEdges.length} edge{routedEdges.length !== 1 ? "s" : ""}
        </span>
        <span className={styles.statsDot}>·</span>
        <span>{workflow.domain}</span>
      </div>

      {/* Legend */}
      <Legend />

      {/* Selected node tooltip */}
      {selectedNode && layoutResult.nodes.has(selectedNode) && (
        <div className={styles.tooltip}>
          <div className={styles.tooltipTitle}>
            {layoutResult.nodes.get(selectedNode)!.label}
          </div>
          <div className={styles.tooltipDesc}>
            {layoutResult.nodes.get(selectedNode)!.description || "No description"}
          </div>
          <div className={styles.tooltipMeta}>
            Layer {layoutResult.nodes.get(selectedNode)!.layer} · Depth{" "}
            {layoutResult.nodes.get(selectedNode)!.depth}
          </div>
        </div>
      )}
    </div>
  );
});
