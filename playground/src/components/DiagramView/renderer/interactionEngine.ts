/* ================================================================
   Interaction Engine — Zero-rerender zoom/pan via CSS transforms
   
   All viewport transforms are applied directly to DOM via refs.
   React state is NOT used for zoom/pan — pure imperative DOM ops.
   
   Features:
   - Mouse wheel zoom (centered on cursor)
   - Mouse drag pan (requestAnimationFrame)
   - Fit-to-view (auto-zoom to fit all content)
   - Center on load
   - Keyboard: +/- zoom, arrow keys pan
   ================================================================ */

export interface ViewportState {
  x: number;
  y: number;
  scale: number;
}

export const MIN_SCALE = 0.15;
export const MAX_SCALE = 3.0;
export const ZOOM_SENSITIVITY = 0.0015;
export const PAN_SPEED = 20;

/* ── Fit-to-view calculator ── */
export function computeFitToView(
  canvasWidth: number,
  canvasHeight: number,
  containerWidth: number,
  containerHeight: number,
  padding = 60,
): ViewportState {
  if (canvasWidth === 0 || canvasHeight === 0) {
    return { x: 0, y: 0, scale: 1 };
  }

  const scaleX = (containerWidth - padding * 2) / canvasWidth;
  const scaleY = (containerHeight - padding * 2) / canvasHeight;
  const scale = Math.min(Math.max(Math.min(scaleX, scaleY), MIN_SCALE), MAX_SCALE);

  const x = (containerWidth - canvasWidth * scale) / 2;
  const y = (containerHeight - canvasHeight * scale) / 2;

  return { x, y, scale };
}

/* ── Apply transform to DOM element ── */
export function applyTransform(
  element: HTMLElement | SVGGElement,
  state: ViewportState,
): void {
  element.style.transform =
    `translate(${state.x}px, ${state.y}px) scale(${state.scale})`;
  element.style.transformOrigin = "0 0";
}

/* ── Zoom centered on a point ── */
export function zoomAtPoint(
  current: ViewportState,
  delta: number,
  pointX: number,
  pointY: number,
): ViewportState {
  const newScale = Math.min(
    MAX_SCALE,
    Math.max(MIN_SCALE, current.scale * (1 - delta * ZOOM_SENSITIVITY)),
  );

  // Adjust translation so the zoom centers on the cursor
  const ratio = newScale / current.scale;
  const newX = pointX - (pointX - current.x) * ratio;
  const newY = pointY - (pointY - current.y) * ratio;

  return { x: newX, y: newY, scale: newScale };
}

/* ── Zoom by fixed step ── */
export function zoomByStep(
  current: ViewportState,
  step: number,
  centerX: number,
  centerY: number,
): ViewportState {
  const newScale = Math.min(
    MAX_SCALE,
    Math.max(MIN_SCALE, current.scale + step),
  );

  const ratio = newScale / current.scale;
  const newX = centerX - (centerX - current.x) * ratio;
  const newY = centerY - (centerY - current.y) * ratio;

  return { x: newX, y: newY, scale: newScale };
}

/* ── Pan by delta ── */
export function panBy(
  current: ViewportState,
  dx: number,
  dy: number,
): ViewportState {
  return { x: current.x + dx, y: current.y + dy, scale: current.scale };
}

/* ── Drag state manager (RAF-based) ── */
export interface DragState {
  active: boolean;
  startX: number;
  startY: number;
  startPanX: number;
  startPanY: number;
}

export function createDragState(): DragState {
  return {
    active: false,
    startX: 0,
    startY: 0,
    startPanX: 0,
    startPanY: 0,
  };
}

export function startDrag(
  drag: DragState,
  viewport: ViewportState,
  clientX: number,
  clientY: number,
): void {
  drag.active = true;
  drag.startX = clientX;
  drag.startY = clientY;
  drag.startPanX = viewport.x;
  drag.startPanY = viewport.y;
}

export function updateDrag(
  drag: DragState,
  clientX: number,
  clientY: number,
): ViewportState | null {
  if (!drag.active) return null;
  return {
    x: drag.startPanX + (clientX - drag.startX),
    y: drag.startPanY + (clientY - drag.startY),
    scale: 0, // caller should preserve scale
  };
}

export function endDrag(drag: DragState): void {
  drag.active = false;
}
