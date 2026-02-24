import { memo, useState, useCallback } from "react";
import type { GenerateResponse } from "../../types/api";
import { DiagramView } from "../DiagramView";
import { JsonView } from "../JsonView";
import { ValidationPanel } from "../ValidationPanel";
import styles from "./OutputWorkspace.module.css";

interface OutputWorkspaceProps {
  response: GenerateResponse | null;
}

type Tab = "diagram" | "json" | "validation";

const TABS: { key: Tab; label: string; icon: string }[] = [
  { key: "diagram", label: "Visual Diagram", icon: "◇" },
  { key: "json", label: "Raw JSON", icon: "{ }" },
  { key: "validation", label: "Validation", icon: "✓" },
];

function OutputWorkspace({ response }: OutputWorkspaceProps) {
  const [activeTab, setActiveTab] = useState<Tab>("diagram");

  const handleTabClick = useCallback((tab: Tab) => {
    setActiveTab(tab);
  }, []);

  /* Badge counts for tabs */
  const issueCount = response?.validation?.issues.length ?? 0;
  const isValid = response?.validation?.is_valid ?? true;

  return (
    <div className={styles.container}>
      {/* Tab bar */}
      <div className={styles.tabBar}>
        {TABS.map((tab) => (
          <button
            key={tab.key}
            className={`${styles.tab} ${
              activeTab === tab.key ? styles.tabActive : ""
            }`}
            onClick={() => handleTabClick(tab.key)}
          >
            <span className={styles.tabIcon}>{tab.icon}</span>
            {tab.label}

            {/* issue count badge on validation tab */}
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

      {/* Tab content */}
      <div className={styles.content}>
        {activeTab === "diagram" && (
          <DiagramView workflow={response?.workflow ?? null} />
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
