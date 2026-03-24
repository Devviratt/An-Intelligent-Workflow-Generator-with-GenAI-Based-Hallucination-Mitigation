import { memo, useCallback, useRef, useState } from "react";
import type { GenerationStatus } from "@/hooks";
import type { DomainInfo, GenerationMode } from "@/types";
import styles from "./ControlPanel.module.css";

interface ControlPanelProps {
  domains: DomainInfo[];
  status: GenerationStatus;
  error: string | null;
  onGenerate: (instruction: string, mode: GenerationMode, domainHint?: string) => void;
  onCancel: () => void;
  onReset: () => void;
}

export const ControlPanel = memo(function ControlPanel({
  domains,
  status,
  error,
  onGenerate,
  onCancel,
  onReset,
}: ControlPanelProps) {
  const [instruction, setInstruction] = useState("");
  const [mode, setMode] = useState<GenerationMode>("workflow");
  const [domainHint, setDomainHint] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const isLoading = status === "loading";

  const handleSubmit = useCallback(() => {
    const trimmed = instruction.trim();
    if (!trimmed || isLoading) return;
    onGenerate(trimmed, mode, domainHint || undefined);
  }, [domainHint, instruction, isLoading, mode, onGenerate]);

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit],
  );

  const handleReset = useCallback(() => {
    setInstruction("");
    setMode("workflow");
    setDomainHint("");
    onReset();
    textareaRef.current?.focus();
  }, [onReset]);

  return (
    <aside className={`glass ${styles.panel}`}>
      <div className={styles.header}>
        <span className={styles.eyebrow}>Prompt Builder</span>
        <h2 className={styles.title}>Describe the workflow you want to visualize</h2>
        <p className={styles.subtitle}>
          Enter an instruction, choose a rendering mode, and generate a graph
          from the backend workflow response.
        </p>
      </div>

      <label className={styles.label}>Instruction</label>
      <textarea
        ref={textareaRef}
        className={styles.textarea}
        placeholder='Example: "Build a CI/CD pipeline with tests, approval, deployment, and rollback branches"'
        value={instruction}
        onChange={(event) => setInstruction(event.target.value)}
        onKeyDown={handleKeyDown}
        rows={7}
        disabled={isLoading}
        spellCheck={false}
      />

      <div className={styles.hintRow}>
        <span className={styles.hint}>Ctrl/Cmd + Enter to generate</span>
        <button
          className={styles.exampleBtn}
          type="button"
          disabled={isLoading}
          onClick={() =>
            setInstruction(
              "Build a CI/CD deployment pipeline with approval, production health checks, and rollback branches",
            )
          }
        >
          Use sample
        </button>
      </div>

      <label className={styles.label}>Mode</label>
      <select
        className={styles.select}
        value={mode}
        onChange={(event) => setMode(event.target.value as GenerationMode)}
        disabled={isLoading}
      >
        <option value="workflow">workflow</option>
        <option value="flowchart">flowchart</option>
      </select>

      <label className={styles.label}>Domain</label>
      <select
        className={styles.select}
        value={domainHint}
        onChange={(event) => setDomainHint(event.target.value)}
        disabled={isLoading}
      >
        <option value="">Auto-detect</option>
        {domains.map((domain) => (
          <option key={domain.domain} value={domain.domain}>
            {domain.display_name}
          </option>
        ))}
      </select>

      <div className={styles.actions}>
        <button
          className={`${styles.btn} ${styles.btnSecondary}`}
          onClick={handleReset}
          disabled={isLoading && !instruction.trim()}
          type="button"
        >
          Reset
        </button>

        {isLoading ? (
          <button
            className={`${styles.btn} ${styles.btnCancel}`}
            onClick={onCancel}
            type="button"
          >
            <span className={styles.spinner} /> Cancel
          </button>
        ) : (
          <button
            className={`${styles.btn} ${styles.btnPrimary}`}
            onClick={handleSubmit}
            disabled={!instruction.trim()}
            type="button"
          >
            Generate Workflow
          </button>
        )}
      </div>

      <div className={styles.infoCard}>
        <h3 className={styles.infoTitle}>What you’ll see</h3>
        <ul className={styles.infoList}>
          <li>Typed nodes for start, process, decision, and end states</li>
          <li>Directed arrows with animated edges and branch labels</li>
          <li>Zoom, pan, fullscreen, SVG export, and validation context</li>
        </ul>
      </div>

      {error && (
        <div className={styles.errorPanel}>
          <span className={styles.errorIcon}>!</span>
          <span>{error}</span>
        </div>
      )}
    </aside>
  );
});
