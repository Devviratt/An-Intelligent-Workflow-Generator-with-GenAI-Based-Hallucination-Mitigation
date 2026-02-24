import { memo, useCallback, useMemo, useState } from "react";
import type { GenerateResponse } from "../../types/api";
import styles from "./JsonView.module.css";

interface JsonViewProps {
  response: GenerateResponse | null;
}

/* ── tiny syntax highlighter ── */
function highlightJson(json: string): string {
  return json.replace(
    /("(?:\\.|[^"\\])*")\s*:/g,
    '<span class="json-key">$1</span>:',
  )
  .replace(
    /:\s*("(?:\\.|[^"\\])*")/g,
    ': <span class="json-string">$1</span>',
  )
  .replace(
    /:\s*(true|false)/g,
    ': <span class="json-bool">$1</span>',
  )
  .replace(
    /:\s*(null)/g,
    ': <span class="json-null">$1</span>',
  )
  .replace(
    /:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g,
    ': <span class="json-number">$1</span>',
  );
}

function JsonView({ response }: JsonViewProps) {
  const [copied, setCopied] = useState(false);

  const raw = useMemo(
    () => (response ? JSON.stringify(response, null, 2) : ""),
    [response],
  );

  const highlighted = useMemo(() => highlightJson(raw), [raw]);

  const lineCount = useMemo(
    () => raw.split("\n").length,
    [raw],
  );

  /* ── actions ── */
  const handleCopy = useCallback(async () => {
    if (!raw) return;
    await navigator.clipboard.writeText(raw);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [raw]);

  const handleDownload = useCallback(() => {
    if (!raw) return;
    const blob = new Blob([raw], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `workflow-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [raw]);

  /* ── empty state ── */
  if (!response) {
    return (
      <div className={styles.empty}>
        <div className={styles.emptyIcon}>{"{ }"}</div>
        <p>Generate a workflow to see the JSON response</p>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      {/* toolbar */}
      <div className={styles.toolbar}>
        <span className={styles.meta}>
          {lineCount} lines &middot;{" "}
          {(new TextEncoder().encode(raw).length / 1024).toFixed(1)} KB
        </span>

        <div className={styles.actions}>
          <button
            className={styles.actionBtn}
            onClick={handleCopy}
            title="Copy to clipboard"
          >
            {copied ? "✓ Copied" : "⎘ Copy"}
          </button>
          <button
            className={styles.actionBtn}
            onClick={handleDownload}
            title="Download JSON"
          >
            ↓ Download
          </button>
        </div>
      </div>

      {/* code block */}
      <div className={styles.codeWrapper}>
        <div className={styles.lineNumbers} aria-hidden>
          {Array.from({ length: lineCount }, (_, i) => (
            <span key={i}>{i + 1}</span>
          ))}
        </div>
        <pre className={styles.code}>
          <code dangerouslySetInnerHTML={{ __html: highlighted }} />
        </pre>
      </div>
    </div>
  );
}

export default memo(JsonView);
