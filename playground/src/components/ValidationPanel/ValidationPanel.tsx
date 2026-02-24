import { memo, useMemo } from "react";
import type {
  ValidationResult,
  ValidationIssue,
  IssueSeverity,
  IssueCategory,
} from "../../types/api";
import styles from "./ValidationPanel.module.css";

interface ValidationPanelProps {
  validation: ValidationResult | null;
}

/* ── severity config ── */
const SEVERITY: Record<IssueSeverity, { label: string; cls: string; icon: string }> = {
  error: { label: "Error", cls: "severityError", icon: "✕" },
  warning: { label: "Warning", cls: "severityWarning", icon: "▲" },
  info: { label: "Info", cls: "severityInfo", icon: "ℹ" },
};

const CATEGORY_LABELS: Record<IssueCategory, string> = {
  schema: "Schema",
  logical: "Logical",
  dependency: "Dependency",
  cycle: "Cycle Detection",
  depth: "Depth Limit",
  grounding: "Grounding",
  structure: "Structure",
  transition: "Transition",
  duplicate: "Duplicate",
  retry: "Retry Policy",
  flowchart: "Flowchart",
  reachability: "Reachability",
};

/* ── issue row ── */
function IssueRow({ issue }: { issue: ValidationIssue }) {
  const sev = SEVERITY[issue.severity];
  return (
    <div className={styles.issueRow}>
      <span className={`${styles.badge} ${styles[sev.cls]}`}>
        <span className={styles.badgeIcon}>{sev.icon}</span>
        {sev.label}
      </span>
      <span className={styles.categoryTag}>
        {CATEGORY_LABELS[issue.category] ?? issue.category}
      </span>
      <span className={styles.issueMessage}>{issue.message}</span>
      {issue.node_id && (
        <code className={styles.nodeRef}>node:{issue.node_id}</code>
      )}
      {issue.edge_id && (
        <code className={styles.nodeRef}>edge:{issue.edge_id}</code>
      )}
    </div>
  );
}

/* ── main component ── */
function ValidationPanel({ validation }: ValidationPanelProps) {
  /* group issues by severity */
  const grouped = useMemo(() => {
    if (!validation) return null;
    const map: Record<IssueSeverity, ValidationIssue[]> = {
      error: [],
      warning: [],
      info: [],
    };
    for (const issue of validation.issues) {
      map[issue.severity].push(issue);
    }
    return map;
  }, [validation]);

  /* empty state */
  if (!validation) {
    return (
      <div className={styles.empty}>
        <div className={styles.emptyIcon}>✓</div>
        <p>Generate a workflow to see validation results</p>
      </div>
    );
  }

  const errors = grouped!.error.length;
  const warnings = grouped!.warning.length;
  const infos = grouped!.info.length;

  return (
    <div className={styles.container}>
      {/* Summary header */}
      <div className={styles.summary}>
        <div
          className={`${styles.verdictBadge} ${
            validation.is_valid ? styles.verdictPass : styles.verdictFail
          }`}
        >
          {validation.is_valid ? "✓ PASS" : "✕ FAIL"}
        </div>

        <div className={styles.counters}>
          {errors > 0 && (
            <span className={`${styles.counter} ${styles.severityError}`}>
              {errors} error{errors !== 1 ? "s" : ""}
            </span>
          )}
          {warnings > 0 && (
            <span className={`${styles.counter} ${styles.severityWarning}`}>
              {warnings} warning{warnings !== 1 ? "s" : ""}
            </span>
          )}
          {infos > 0 && (
            <span className={`${styles.counter} ${styles.severityInfo}`}>
              {infos} info
            </span>
          )}
          {validation.issues.length === 0 && (
            <span className={styles.counterClean}>No issues found</span>
          )}
        </div>

        <div className={styles.stats}>
          <span>{validation.nodes_validated} nodes</span>
          <span>{validation.edges_validated} edges</span>
          <span>{validation.checks_performed.length} checks</span>
        </div>
      </div>

      {/* Checks performed */}
      <div className={styles.checksSection}>
        <h4 className={styles.sectionTitle}>Checks Performed</h4>
        <div className={styles.checksList}>
          {validation.checks_performed.map((check) => (
            <span key={check} className={styles.checkChip}>
              {check}
            </span>
          ))}
        </div>
      </div>

      {/* Issues list */}
      {validation.issues.length > 0 && (
        <div className={styles.issuesSection}>
          <h4 className={styles.sectionTitle}>
            Issues ({validation.issues.length})
          </h4>
          <div className={styles.issuesList}>
            {/* errors first, then warnings, then info */}
            {grouped!.error.map((issue, i) => (
              <IssueRow key={`e-${i}`} issue={issue} />
            ))}
            {grouped!.warning.map((issue, i) => (
              <IssueRow key={`w-${i}`} issue={issue} />
            ))}
            {grouped!.info.map((issue, i) => (
              <IssueRow key={`i-${i}`} issue={issue} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default memo(ValidationPanel);
