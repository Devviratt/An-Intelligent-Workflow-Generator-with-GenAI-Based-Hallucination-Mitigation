import { ControlPanel } from "@/components/ControlPanel";
import { OutputWorkspace } from "@/components/OutputWorkspace";
import { useWorkflowGenerator } from "@/hooks";
import styles from "./App.module.css";

export default function App() {
  const {
    status,
    response,
    error,
    domains,
    generate,
    cancel,
    reset,
  } = useWorkflowGenerator();

  return (
    <div className={styles.layout}>
      <header className={styles.header}>
        <div className={styles.brandBlock}>
          <div className={styles.brand}>
            <span className={styles.logo}>WF</span>
            <div>
              <h1 className={styles.title}>Intelligent Workflow Generator</h1>
              <p className={styles.subtitle}>Visual Workflow Engine</p>
            </div>
          </div>
          <p className={styles.tagline}>
            Turn natural-language instructions into a structured workflow graph
            with branching logic, validation details, and export-ready visuals.
          </p>
        </div>

        <div className={styles.statusPill} data-state={status}>
          {status === "loading" ? "Generating..." : "Backend Ready"}
        </div>
      </header>

      <main className={styles.main}>
        <aside className={styles.sidebar}>
          <ControlPanel
            domains={domains}
            status={status}
            error={error}
            onGenerate={generate}
            onCancel={cancel}
            onReset={reset}
          />
        </aside>

        <section className={styles.workspace}>
          <OutputWorkspace
            response={response}
            status={status}
            error={error}
          />
        </section>
      </main>
    </div>
  );
}
