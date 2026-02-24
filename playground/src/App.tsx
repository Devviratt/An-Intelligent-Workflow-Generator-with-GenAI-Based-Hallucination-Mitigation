/* ================================================================
   App — root layout component.

   Two-panel layout:
     Left  → ControlPanel  (instruction, mode, domain, generate)
     Right → OutputWorkspace (diagram, JSON, validation tabs)
   ================================================================ */

import { useWorkflowGenerator } from "@/hooks";
import { ControlPanel } from "@/components/ControlPanel";
import { OutputWorkspace } from "@/components/OutputWorkspace";
import styles from "./App.module.css";

export default function App() {
  const {
    status,
    response,
    error,
    domains,
    generate,
    cancel,
  } = useWorkflowGenerator();

  return (
    <div className={styles.layout}>
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.brand}>
          <span className={styles.logo}>◇</span>
          <h1 className={styles.title}>Workflow Playground</h1>
          <span className={styles.version}>v1</span>
        </div>
        <p className={styles.subtitle}>
          AI-powered workflow &amp; flowchart generation with hallucination
          mitigation
        </p>
      </header>

      {/* Main content */}
      <main className={styles.main}>
        <aside className={styles.sidebar}>
          <ControlPanel
            domains={domains}
            status={status}
            error={error}
            onGenerate={generate}
            onCancel={cancel}
          />
        </aside>

        <section className={styles.workspace}>
          <OutputWorkspace response={response} />
        </section>
      </main>
    </div>
  );
}
