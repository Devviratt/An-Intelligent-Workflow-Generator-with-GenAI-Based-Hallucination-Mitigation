/* ================================================================
   ControlPanel — left sidebar with instruction input, mode toggle,
   domain selector, and generate button.
   ================================================================ */

import { memo, useCallback, useRef, useState } from "react";
import type { DomainInfo, GenerationMode } from "@/types";
import type { GenerationStatus } from "@/hooks";
import styles from "./ControlPanel.module.css";

interface ControlPanelProps {
  domains: DomainInfo[];
  status: GenerationStatus;
  error: string | null;
  onGenerate: (instruction: string, mode: GenerationMode, domainHint?: string) => void;
  onCancel: () => void;
}

export const ControlPanel = memo(function ControlPanel({
  domains,
  status,
  error,
  onGenerate,
  onCancel,
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
  }, [instruction, mode, domainHint, isLoading, onGenerate]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit],
  );

  return (
    <aside className={`glass ${styles.panel}`}>
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.logo}>
          <span className={styles.logoIcon}>⚡</span>
          <div>
            <h1 className={styles.title}>Workflow Generator</h1>
            <p className={styles.subtitle}>API Playground</p>
          </div>
        </div>
      </div>

      {/* Instruction */}
      <label className={styles.label}>Instruction</label>
      <textarea
        ref={textareaRef}
        className={styles.textarea}
        placeholder="Describe the workflow you want to generate…"
        value={instruction}
        onChange={(e) => setInstruction(e.target.value)}
        onKeyDown={handleKeyDown}
        rows={5}
        disabled={isLoading}
        spellCheck={false}
      />
      <span className={styles.hint}>⌘ + Enter to generate</span>

      {/* Mode toggle */}
      <label className={styles.label}>Mode</label>
      <div className={styles.toggleGroup}>
        <button
          className={`${styles.toggleBtn} ${mode === "workflow" ? styles.active : ""}`}
          onClick={() => setMode("workflow")}
          disabled={isLoading}
          type="button"
        >
          Workflow
        </button>
        <button
          className={`${styles.toggleBtn} ${mode === "flowchart" ? styles.active : ""}`}
          onClick={() => setMode("flowchart")}
          disabled={isLoading}
          type="button"
        >
          Flowchart
        </button>
      </div>

      {/* Domain hint */}
      <label className={styles.label}>Domain (optional)</label>
      <select
        className={styles.select}
        value={domainHint}
        onChange={(e) => setDomainHint(e.target.value)}
        disabled={isLoading}
      >
        <option value="">Auto-detect</option>
        {domains.map((d) => (
          <option key={d.domain} value={d.domain}>
            {d.display_name}
          </option>
        ))}
      </select>

      {/* Actions */}
      <div className={styles.actions}>
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
            Generate
          </button>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className={styles.errorPanel}>
          <span className={styles.errorIcon}>⚠</span>
          <span>{error}</span>
        </div>
      )}
    </aside>
  );
});
