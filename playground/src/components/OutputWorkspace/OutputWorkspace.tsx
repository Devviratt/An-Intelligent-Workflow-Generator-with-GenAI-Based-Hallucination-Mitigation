import { memo, useCallback, useState } from "react";
import type { GenerationStatus } from "@/hooks";
import type { GenerateResponse } from "../../types/api";
import { DiagramView } from "../DiagramView";
import { JsonView } from "../JsonView";
import { ValidationPanel } from "../ValidationPanel";
import styles from "./OutputWorkspace.module.css";

interface OutputWorkspaceProps {
  response: GenerateResponse | null;
  status: GenerationStatus;
  error: string | null;
}

type Tab = "diagram" | "json" | "validation";

const TABS: { key: Tab; label: string; icon: string }[] = [
  { key: "diagram", label: "Visual Graph", icon: "◇" },
  { key: "json", label: "Response JSON", icon: "{ }" },
  { key: "validation", label: "Validation", icon: "✓" },
];

function OutputWorkspace({ response, status, error }: OutputWorkspaceProps) {
  const [activeTab, setActiveTab] = useState<Tab>("diagram");

  const handleTabClick = useCallback((tab: Tab) => {
    setActiveTab(tab);
  }, []);

  const issueCount = response?.validation?.issues.length ?? 0;
  const isValid = response?.validation?.is_valid ?? true;
  const invalidNodeIds = new Set(
    (response?.validation?.issues ?? [])
      .filter((issue) => issue.severity === "error" && issue.node_id)
      .map((issue) => issue.node_id as string),
  );

  return (
    <div className={styles.container}>
      <div className={styles.heroBar}>
        <div>
          <h2 className={styles.heroTitle}>Workflow Graph Canvas</h2>
          <p className={styles.heroSubtitle}>
            Visualize nodes, branching logic, edge labels, and validation state
            in one place.
          </p>
        </div>

        {response?.workflow && (
          <div className={styles.metaPills}>
            <span className={styles.metaPill}>{response.workflow.domain}</span>
            <span className={styles.metaPill}>
              {response.workflow.is_flowchart ? "flowchart" : "workflow"}
            </span>
          </div>
        )}
      </div>

      <div className={styles.tabBar}>
        {TABS.map((tab) => (
          <button
            key={tab.key}
            className={`${styles.tab} ${
              activeTab === tab.key ? styles.tabActive : ""
            }`}
            onClick={() => handleTabClick(tab.key)}
            type="button"
          >
            <span className={styles.tabIcon}>{tab.icon}</span>
            {tab.label}

            {tab.key === "validation" && response && (
              <span
                className={`${styles.tabBadge} ${
                  isValid ? styles.tabBadgePass : styles.tabBadgeFail
                }`}
              >
                {issueCount === 0 ? "✓" : issueCount}
              </span>
            )}
          </button>
        ))}
      </div>

      <div className={styles.content}>
        {status === "loading" && (
          <div className={styles.loadingOverlay}>
            <div className={styles.loadingSpinner} />
            <div className={styles.loadingText}>Generating workflow graph...</div>
          </div>
        )}

        {error && !response && (
          <div className={styles.errorBanner}>
            <strong>Request failed.</strong> {error}
          </div>
        )}

        {activeTab === "diagram" && (
          <DiagramView
            workflow={response?.workflow ?? null}
            invalidNodeIds={invalidNodeIds}
          />
        )}
        {activeTab === "json" && <JsonView response={response} />}
        {activeTab === "validation" && (
          <ValidationPanel validation={response?.validation ?? null} />
        )}
      </div>
    </div>
  );
}

export default memo(OutputWorkspace);
