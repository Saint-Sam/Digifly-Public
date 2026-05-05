from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .cache_identity import format_impact_summary
from .controls import CONTROL_SPECS, ControlSpec, control_by_key, default_state, sections_for_state, specs_in_section
from .mutation_launcher import build_mutation_launch_plan, launch_mutation_app, log_tail
from .presets import PRESETS, apply_preset, get_preset, iter_notes, preset_options
from .runner import build_execution_plan, execute_plan
from .validation import validate_state

try:
    import ipywidgets as widgets
    from IPython.display import Markdown, display
except Exception:  # pragma: no cover - depends on notebook runtime
    widgets = None
    Markdown = None
    display = None


class Phase2WorkbenchUI:
    def __init__(self, phase2_root: str | Path):
        if widgets is None or display is None:
            raise RuntimeError("ipywidgets and IPython.display are required for the Phase 2 workbench notebook.")
        self.phase2_root = Path(phase2_root).expanduser().resolve()
        self.spec_by_key = control_by_key()
        self.widgets: Dict[str, widgets.Widget] = {}
        self.rows: Dict[str, widgets.Widget] = {}
        self.sections: Dict[str, widgets.Widget] = {}
        self.current_state = default_state()
        self.last_run_result: Dict[str, Any] | None = None
        self.last_run_dir: Path | None = None

        self.preset_dropdown = widgets.Dropdown(options=preset_options(), value=PRESETS[0].slug, description="Preset")
        self.apply_preset_button = widgets.Button(description="Apply Preset", button_style="")
        self.preview_button = widgets.Button(description="Preview Plan", button_style="info")
        self.validate_button = widgets.Button(description="Validate", button_style="")
        self.bundle_button = widgets.Button(description="Write Bundle", button_style="")
        self.run_button = widgets.Button(description="Run", button_style="success")
        self.mutation_button = widgets.Button(
            description="Open Mutation App",
            button_style="warning",
            disabled=True,
            tooltip="Launch the morphology mutation app with the most recent completed workbench run.",
        )
        self.status_html = widgets.HTML()
        self.notes_html = widgets.HTML()
        self.output = widgets.Output(layout=widgets.Layout(border="1px solid #ddd", padding="8px"))
        self.form = widgets.VBox()

        self._build_form()
        self._wire_events()
        self._apply_preset(self.preset_dropdown.value)

    def display(self) -> None:
        header = widgets.VBox(
            [
                widgets.HTML("<h2>Digifly Phase 2 Workbench</h2>"),
                widgets.HTML(
                    "<p>Notebook-first control surface for the public Phase 2 runner. "
                    "Use presets for clean starting points, then adjust fields or the advanced JSON boxes.</p>"
                ),
            ]
        )
        toolbar = widgets.HBox(
            [
                self.preset_dropdown,
                self.apply_preset_button,
                self.preview_button,
                self.validate_button,
                self.bundle_button,
                self.run_button,
                self.mutation_button,
            ]
        )
        display(widgets.VBox([header, toolbar, self.status_html, self.notes_html, self.form, self.output]))

    def _build_form(self) -> None:
        for spec in CONTROL_SPECS:
            widget = self._make_widget(spec)
            self.widgets[spec.key] = widget
            label = widgets.HTML(f"<b>{spec.label}</b><br><span style='color:#555'>{spec.help_text}</span>")
            row = widgets.VBox([label, widget], layout=widgets.Layout(padding="6px 0"))
            self.rows[spec.key] = row

        section_widgets = []
        for section in sections_for_state(self.current_state):
            box = widgets.VBox([], layout=widgets.Layout(border="1px solid #eee", padding="10px", margin="0 0 10px 0"))
            self.sections[section] = box
            section_widgets.append(widgets.VBox([widgets.HTML(f"<h3>{section}</h3>"), box]))
        self.form.children = tuple(section_widgets)
        self._refresh_form()

    def _wire_events(self) -> None:
        self.apply_preset_button.on_click(lambda _: self._apply_preset(self.preset_dropdown.value))
        self.preview_button.on_click(lambda _: self._preview())
        self.validate_button.on_click(lambda _: self._validate())
        self.bundle_button.on_click(lambda _: self._write_bundle_only())
        self.run_button.on_click(lambda _: self._run())
        self.mutation_button.on_click(lambda _: self._launch_mutation_app())
        self.widgets["runner_kind"].observe(lambda _: self._refresh_form(), names="value")
        self.widgets["mode"].observe(lambda _: self._refresh_form(), names="value")

    def _make_widget(self, spec: ControlSpec):
        layout = widgets.Layout(width="100%")
        if spec.control_type == "choice":
            return widgets.Dropdown(options=list(spec.choices), value=spec.default, layout=layout)
        if spec.control_type == "bool":
            return widgets.Checkbox(value=bool(spec.default), indent=False, layout=layout)
        if spec.control_type == "int":
            return widgets.IntText(value=int(spec.default), layout=layout)
        if spec.control_type == "float":
            return widgets.FloatText(value=float(spec.default), layout=layout)
        if spec.control_type == "textarea":
            return widgets.Textarea(value=str(spec.default), layout=widgets.Layout(width="100%", height="110px"))
        return widgets.Text(value=str(spec.default), layout=layout)

    def _state_from_widgets(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        for key, widget in self.widgets.items():
            state[key] = widget.value
        return state

    def _apply_preset(self, slug: str) -> None:
        state = apply_preset(slug)
        self.current_state = state
        for key, value in state.items():
            widget = self.widgets[key]
            widget.value = value
        notes = list(iter_notes(slug))
        if notes:
            self.notes_html.value = "<ul>" + "".join(f"<li>{note}</li>" for note in notes) + "</ul>"
        else:
            self.notes_html.value = ""
        self.status_html.value = f"<p><b>Preset loaded:</b> {get_preset(slug).label}</p>"
        self._refresh_form()

    def _refresh_form(self) -> None:
        state = self._state_from_widgets()
        current_sections = sections_for_state(state)
        rebuilt_children = []
        for section in current_sections:
            visible_rows = [self.rows[spec.key] for spec in specs_in_section(section, state)]
            if section not in self.sections:
                self.sections[section] = widgets.VBox([], layout=widgets.Layout(border="1px solid #eee", padding="10px", margin="0 0 10px 0"))
            self.sections[section].children = tuple(visible_rows)
            rebuilt_children.append(widgets.VBox([widgets.HTML(f"<h3>{section}</h3>"), self.sections[section]]))
        self.form.children = tuple(rebuilt_children)

    def _preview(self) -> None:
        state = self._state_from_widgets()
        with self.output:
            self.output.clear_output()
            try:
                plan = build_execution_plan(state, preset_slug=self.preset_dropdown.value)
                print(plan.description)
                print("")
                print("Impact summary:")
                print(format_impact_summary(plan.impact_summary))
                print("")
                print("Payload preview:")
                print(json.dumps(plan.payload, indent=2, default=str))
                self.status_html.value = "<p><b>Preview ready.</b></p>"
            except Exception as exc:
                print(f"Preview failed: {exc}")
                self.status_html.value = f"<p><b>Preview failed.</b> {exc}</p>"

    def _validate(self) -> None:
        state = self._state_from_widgets()
        report = validate_state(state)
        with self.output:
            self.output.clear_output()
            print(report.format())
        if report.ok:
            self.status_html.value = "<p><b>Validation OK.</b></p>"
        else:
            self.status_html.value = "<p><b>Validation failed.</b> See output below.</p>"

    def _write_bundle_only(self) -> None:
        state = self._state_from_widgets()
        with self.output:
            self.output.clear_output()
            try:
                plan = build_execution_plan(state, preset_slug=self.preset_dropdown.value)
                from .artifacts import prepare_workbench_bundle

                bundle_dir = prepare_workbench_bundle(
                    self.phase2_root,
                    run_id=plan.run_id,
                    preset_slug=plan.preset_slug,
                    state=state,
                    plan=plan.to_dict(),
                )
                print(f"Bundle written to: {bundle_dir}")
                self.status_html.value = f"<p><b>Bundle written.</b> {bundle_dir}</p>"
            except Exception as exc:
                print(f"Bundle write failed: {exc}")
                self.status_html.value = f"<p><b>Bundle write failed.</b> {exc}</p>"

    def _run(self) -> None:
        state = self._state_from_widgets()
        with self.output:
            self.output.clear_output()
            try:
                result = execute_plan(state, preset_slug=self.preset_dropdown.value, phase2_root=self.phase2_root)
                print(json.dumps(result, indent=2, default=str))
                self.last_run_result = result
                self.last_run_dir = _result_run_dir(result)
                self.mutation_button.disabled = self.last_run_dir is None
                self._display_voltage_traces(result)
                if self.last_run_dir is not None:
                    print(f"\nMutation app ready for run: {self.last_run_dir}")
                    self.status_html.value = "<p><b>Run completed.</b> Voltage traces loaded; mutation app launcher is ready.</p>"
                else:
                    self.status_html.value = "<p><b>Run completed.</b> See output below.</p>"
            except Exception as exc:
                print(f"Run failed: {exc}")
                self.status_html.value = f"<p><b>Run failed.</b> {exc}</p>"

    def _launch_mutation_app(self) -> None:
        state = self._state_from_widgets()
        with self.output:
            if self.last_run_dir is None:
                print("No completed workbench run is available yet. Run a simulation first.")
                self.status_html.value = "<p><b>Mutation app not launched.</b> Run a simulation first.</p>"
                return
            try:
                plan = build_mutation_launch_plan(
                    state,
                    phase2_root=self.phase2_root,
                    flow_run_dir=self.last_run_dir,
                )
                print("\nLaunching morphology mutation app with latest workbench run:")
                print(f"  run_dir: {plan.flow_run_dir}")
                print(f"  swc_dir: {plan.swc_dir}")
                print(f"  neuron_ids: {plan.neuron_ids}")
                if plan.warning:
                    print(f"\nWarning: {plan.warning}")
                print("\nCommand:")
                print(plan.command_text())
                launch = launch_mutation_app(plan)
                if launch.get("blocked"):
                    print("\nDesktop launch skipped.")
                    print(launch.get("reason") or "Current Python session cannot launch the desktop app.")
                    print("\nRun the printed command from a desktop Python session to open the current PyVista app.")
                    self.status_html.value = "<p><b>Mutation app not launched.</b> Desktop display is unavailable in this kernel.</p>"
                elif launch["returncode"] is None:
                    print(f"\nLaunched morphology mutation app (PID={launch['pid']}).")
                    print(f"Launcher log: {launch['log_path']}")
                    self.status_html.value = "<p><b>Mutation app launched.</b> Latest workbench run is connected as flow input.</p>"
                else:
                    print(f"\nMutation app exited early with code {launch['returncode']}.")
                    print(f"Launcher log: {launch['log_path']}")
                    print("--- launcher log tail ---")
                    print(log_tail(launch["log_path"]))
                    self.status_html.value = "<p><b>Mutation app launch failed.</b> See output below.</p>"
            except Exception as exc:
                print(f"Mutation app launch failed: {exc}")
                self.status_html.value = f"<p><b>Mutation app launch failed.</b> {exc}</p>"

    def _display_voltage_traces(self, result: Dict[str, Any]) -> None:
        run_dir = _result_run_dir(result)
        if run_dir is None:
            print("\nVoltage traces: no run output directory found in result.")
            return
        records_path = run_dir / "records.csv"
        if not records_path.exists():
            print(f"\nVoltage traces: records.csv not found at {records_path}")
            return

        try:
            import matplotlib.pyplot as plt
            import pandas as pd
        except Exception as exc:
            print(f"\nVoltage traces: pandas/matplotlib unavailable: {exc}")
            return

        try:
            df = pd.read_csv(records_path)
        except Exception as exc:
            print(f"\nVoltage traces: could not read {records_path}: {exc}")
            return

        time_col = _find_time_column(df)
        trace_cols = [str(col) for col in df.columns if str(col).endswith("_soma_v")]
        if time_col is None or not trace_cols:
            print(f"\nVoltage traces: no soma voltage traces found in {records_path}")
            return

        max_traces = 12
        plot_cols = trace_cols[:max_traces]
        ax = df.plot(x=time_col, y=plot_cols, figsize=(10, 4), linewidth=1.4)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Soma voltage (mV)")
        ax.set_title(f"Voltage traces: {run_dir.name}")
        ax.grid(True, alpha=0.25)
        if len(trace_cols) > max_traces:
            ax.text(
                0.01,
                0.98,
                f"Showing {max_traces} of {len(trace_cols)} traces",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
            )
        plt.tight_layout()
        display(ax.figure)
        plt.close(ax.figure)


def _result_run_dir(result: Dict[str, Any]) -> Path | None:
    if not isinstance(result, dict):
        return None
    payload = result.get("result")
    if not isinstance(payload, dict):
        return None
    for key in ("output_dir", "baseline_out_dir", "metadata_dir", "project_root"):
        value = payload.get(key)
        if value:
            return Path(str(value)).expanduser().resolve()
    return None


def _find_time_column(df: Any) -> str | None:
    for name in ("t_ms", "time_ms", "time", "t"):
        if name in df.columns:
            return str(name)
    return str(df.columns[0]) if len(df.columns) else None


def launch_workbench(phase2_root: str | Path):
    ui = Phase2WorkbenchUI(phase2_root=phase2_root)
    ui.display()
    return ui
