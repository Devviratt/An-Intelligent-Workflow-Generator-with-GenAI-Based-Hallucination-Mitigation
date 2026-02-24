/* ================================================================
   Crossing Minimizer — Barycenter heuristic
   
   Reduces edge crossings between adjacent layers by reordering
   nodes within each layer according to the barycenter (average
   position) of their connected nodes in the adjacent layer.
   
   Runs multiple sweeps (down then up) for convergence.
   Deterministic and O(V + E) per sweep.
   ================================================================ */

const MAX_SWEEPS = 4;

/* ── Barycenter for a single layer ── */
function computeBarycenter(
  layer: string[],
  adjacentLayer: string[],
  edges: Map<string, string[]>,
  direction: "down" | "up",
): Map<string, number> {
  const posMap = new Map<string, number>();
  for (let i = 0; i < adjacentLayer.length; i++) {
    posMap.set(adjacentLayer[i]!, i);
  }

  const barycenters = new Map<string, number>();

  for (const nid of layer) {
    const neighbors = direction === "down"
      ? (edges.get(nid) ?? []).filter((t) => posMap.has(t))
      : Array.from(edges.entries())
          .filter(([, targets]) => targets.includes(nid))
          .map(([src]) => src)
          .filter((s) => posMap.has(s));

    if (neighbors.length === 0) {
      barycenters.set(nid, layer.indexOf(nid));
    } else {
      const sum = neighbors.reduce(
        (acc, nb) => acc + (posMap.get(nb) ?? 0),
        0,
      );
      barycenters.set(nid, sum / neighbors.length);
    }
  }

  return barycenters;
}

/* ── Sort layer by barycenter values ── */
function sortByBarycenter(
  layer: string[],
  barycenters: Map<string, number>,
): string[] {
  return [...layer].sort((a, b) => {
    const ba = barycenters.get(a) ?? 0;
    const bb = barycenters.get(b) ?? 0;
    if (ba !== bb) return ba - bb;
    return a.localeCompare(b); // deterministic tiebreak
  });
}

/* ── Count crossings between two adjacent layers ── */
function countCrossings(
  upperLayer: string[],
  lowerLayer: string[],
  edges: Map<string, string[]>,
): number {
  const lowerPos = new Map<string, number>();
  for (let i = 0; i < lowerLayer.length; i++) {
    lowerPos.set(lowerLayer[i]!, i);
  }

  // Collect edge positions (upper_pos, lower_pos)
  const edgePositions: [number, number][] = [];
  for (let ui = 0; ui < upperLayer.length; ui++) {
    const nid = upperLayer[ui]!;
    for (const target of edges.get(nid) ?? []) {
      const li = lowerPos.get(target);
      if (li !== undefined) {
        edgePositions.push([ui, li]);
      }
    }
  }

  // Count inversions (crossings)
  let crossings = 0;
  for (let i = 0; i < edgePositions.length; i++) {
    for (let j = i + 1; j < edgePositions.length; j++) {
      const [u1, l1] = edgePositions[i]!;
      const [u2, l2] = edgePositions[j]!;
      if ((u1 - u2) * (l1 - l2) < 0) {
        crossings++;
      }
    }
  }

  return crossings;
}

/* ================================================================
   PUBLIC API
   ================================================================ */
export function minimizeCrossings(
  layers: string[][],
  forwardEdges: Map<string, string[]>,
): string[][] {
  if (layers.length <= 1) return layers;

  let best = layers.map((l) => [...l]);
  let bestCrossings = Infinity;

  // Compute initial crossing count
  for (let i = 0; i < best.length - 1; i++) {
    bestCrossings += countCrossings(best[i]!, best[i + 1]!, forwardEdges);
  }

  const current = best.map((l) => [...l]);

  for (let sweep = 0; sweep < MAX_SWEEPS; sweep++) {
    // Down sweep
    for (let i = 1; i < current.length; i++) {
      const bary = computeBarycenter(
        current[i]!,
        current[i - 1]!,
        forwardEdges,
        "down",
      );
      current[i] = sortByBarycenter(current[i]!, bary);
    }

    // Up sweep
    for (let i = current.length - 2; i >= 0; i--) {
      const bary = computeBarycenter(
        current[i]!,
        current[i + 1]!,
        forwardEdges,
        "up",
      );
      current[i] = sortByBarycenter(current[i]!, bary);
    }

    // Count crossings
    let totalCrossings = 0;
    for (let i = 0; i < current.length - 1; i++) {
      totalCrossings += countCrossings(
        current[i]!,
        current[i + 1]!,
        forwardEdges,
      );
    }

    if (totalCrossings < bestCrossings) {
      bestCrossings = totalCrossings;
      for (let i = 0; i < current.length; i++) {
        best[i] = [...current[i]!];
      }
    }

    // Converged — no crossings
    if (bestCrossings === 0) break;
  }

  return best;
}
