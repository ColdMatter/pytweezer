---
name: pytweezer-gui-design
description: Visual and UX design for pytweezer's PyQt6 GUIs — how a tab, panel, dialog or applet should be laid out, styled and worded so a physicist can read it at a glance mid-experiment. Use whenever building a new panel/tab/applet/dialog, or when the user says a GUI is hard to read, cluttered, confusing, "everything looks the same", ugly, cramped or wants it tidied, restyled, made clearer or more usable; when choosing colours, buttons, forms, tables, status indicators or labels; when editing pytweezer/GUI/theme.py or any QSS; or when someone asks for a Refresh button. Use alongside pytweezer-gui-internals (shell mechanics) and add-applet (stream viewers) — they say how the plumbing works, this says how it should look and behave.
---

# Designing pytweezer GUIs

The people using these screens are physicists running an experiment: often in a
dark lab, glancing over from the optics table, needing to know *what is running,
what is wrong, and what they can do* in a second. Decoration doesn't help them;
legible structure and honest state do. Every choice below exists to serve that.

## Work in a loop: look, plan, build, look again

You can't judge a layout from code. Always screenshot (see *Seeing it*), and:

1. **Diagnose** the current screen in terms of the principles below — not "it
   looks bad" but "the list and the editor share one grey, and Submit is
   indistinguishable from Defaults".
2. **Plan** in a few lines: the regions and which kind each is, the one primary
   action, which actions are destructive, what updates live, the copy for
   titles/hints/empty states.
3. **Check the plan against the generic default.** If it would come out the same
   for any panel (every block a rounded card, a coloured accent sprinkled
   everywhere), it isn't a decision yet — tie it to this panel's job.
4. **Build, screenshot, critique, fix.** Expect two or three rounds; the first
   pass always has something floating, stretched, truncated or cramped.

## Principles

**1. Depth encodes role.** Use `Region` from `pytweezer/GUI/components.py`:
- `Region("well", title, hint)`: recessed, for things you *read and pick from*
  (lists, tables, logs, plots).
- `Region("sheet")`: raised with an accent edge, for the one place you *edit*.
  At most one per panel; if everything is raised, nothing is.
- Every region has a title. The hint (muted, beside it) either says what to do
  ("pick one to edit") or summarises live state ("task 2 running, 2 waiting").
- Leave the window colour showing between regions (splitter handles 8 px,
  transparent until hovered); the gaps are what separate them.

**2. Spend the accent once.** The blue (`#5b8def`) means "this is the action /
this is selected / this is focused". Give the action the panel exists for
`objectName("PrimaryButton")`, and put its consequence beside it ("21 points"
next to Submit). Don't use blue for decoration.

**3. Actions are grouped by what they act on.**
- Order buttons by target (the running task | waiting tasks | any task), with
  spacing between groups rather than one undifferentiated row.
- Actions that lose work or data get `objectName("DangerButton")` and sit apart,
  at the far right. Confirm the irreversible ones (`QMessageBox.question`).
- Disable buttons that don't apply to the current selection rather than
  erroring after a click; the theme makes disabled buttons look disabled.

**4. Fields have readable widths; slack goes somewhere useful.**
- Fixed widths for values (about 240 px; 110 px inside compact rows), never
  stretched across a 1,900 px window. Forms use
  `QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint`.
- Give spare width to a trailing stretch column, or to the user's own free text
  (a label/notes column), not to identifiers.
- Units sit right after the value, in display units (SI internally; see
  `Number(unit=, scale=)`). Use `CompactDoubleSpinBox` so 0.2 isn't 0.200000.
- Rows read left to right in fixed columns, so the eye can run down them:
  label | value + unit | per-row toggle.

**5. State is colour *and* text.** Use the `STATE_STYLE` vocabulary in
`theme.py` (running/starting/stopped/crashed…), so a state reads the same in
every panel. In item views use `status_icon(state)` beside the status text.
Shade *rows* by lifecycle (the active item tinted, finished ones recessed) so
the eye finds what matters. Never encode state by colour alone.

**6. Live, not Refresh.** If the user has to press Refresh, the panel is lying
until they do. Follow the server's PUB feed (`ExperimentFeed`,
`DeviceStatusClient`) or poll on a `QTimer`, and:
- update in place: keep selection, expansion, scroll position and combo-box
  choices; don't `clear()` and rebuild;
- poll only while visible (`if self.isVisible()`, refresh in `showEvent`),
  since sources may be network shares;
- re-read only what can still change (a finished file never does).

**7. Words do one job each.**
- Sentence case; no ALL-CAPS labels.
- Name things in the user's terms ("Experiments", "Queue", "note saved with the
  measurement"), not the code's.
- Buttons say what happens ("Submit", "Resubmit with these arguments", "Edit as
  new").
- An empty state says what to do ("Pick an experiment on the left to edit it").
- An error says what failed and how to fix it ("Submitting: the Experiment
  Manager is not responding").
- Tooltips carry the detail the label can't ("Kill the task now. A device call
  already in progress still completes").

**8. Say when it isn't real.** If a panel can show simulated or stale data,
say so prominently (the orange `SimulationBanner`), not in a log line.

## How styling works here

All styling is in `DARK_STYLESHEET` in `pytweezer/GUI/theme.py`, applied
app-wide. Widgets opt in with an `objectName` or a dynamic property
(`role`, `state`, `kind`). **Don't call `setStyleSheet` on individual widgets**
(the few that do are legacy): inline styles can't be themed and they fight the
stylesheet's specificity. Add a rule to `theme.py` instead, scoped by
`objectName`. Change a property with `set_state(widget, state)` from
`components.py`, which re-polishes; Qt doesn't restyle on property changes by
itself.

Available hooks:

| Hook | Use |
| --- | --- |
| `QFrame#Region[kind="well"/"sheet"]` | regions (via `Region`) |
| `QLabel[role="regionTitle"/"regionHint"/"experimentTitle"/"heading"]` | titles, muted hints |
| `QLabel#StatusLabel[state=…]`, `QLabel#SimulationBanner` | status line, simulation tag |
| `QPushButton#PrimaryButton`, `#DangerButton`, `#ToggleButton[kind="start"/"stop"]` | actions |
| `QFrame#ProcessTile[state=…]` | process rows (managed_panel) |
| `QGroupBox#EditorGroup`, `QToolButton#ScanToggle` | grouped form sections, checkable toggles |

Palette (extend with these; don't introduce near-duplicates):

| Token | Hex |
| --- | --- |
| window | `#1b1c22` |
| well | `#141519` |
| sheet / panel | `#262730` / `#24252c` |
| field on sheet | `#30313b` |
| edges | `#2c2d35`, `#33343d`, `#44454f` |
| text / muted | `#e6e6e6` / `#8d8e99` |
| accent | `#5b8def` |
| running / warning / error | `#2ecc71` / `#f5a623` / `#e74c3c` |

### Qt and QSS traps (each one cost a round of screenshots)

- **Specificity:** IDs, then attributes and pseudo-states, then types are
  counted. `QFrame#Region[kind="sheet"] QLineEdit` beats `QLineEdit`, and also
  beats `QLineEdit:focus` for any property both set. Override at equal or
  higher specificity, later in the sheet.
- **Item colours are overridden.** The theme's `QTreeView::item { color }` rule
  wins over `item.setForeground()`. Use icons (`status_icon`), item
  backgrounds, or fonts instead.
- **`QScrollArea.setWidget()` turns on `autoFillBackground`,** so the scrolled
  content paints the window colour inside a region. Give the scroll area and
  its widget object names and a `background: transparent` rule (see
  `#EditorScroll`, `#ArgumentEditor`). Plain `QWidget` containers are already
  transparent. `QGroupBox`, `QStackedWidget` and `QCheckBox` need the rule
  inside regions.
- **`ResizeToContents` packs cells edge to edge.** Add `::item` padding for that
  table.
- **pyqtgraph** adds SI prefixes to axis units. If values are already in display
  units, call `axis.enableAutoSIPrefix(False)`, or µs becomes "kµs". Embedded
  plots need `background=PLOT_BACKGROUND` (only applets get `apply_theme`'s
  pyqtgraph defaults).
- **`app.quit()` closes every window in Qt 6.** In a server GUI that stops every
  server it started. Pump events with a local `QEventLoop` in scripts and tests.
- **The font is Segoe UI, which isn't on Linux,** where a fallback is
  substituted. Check sizes on the lab PCs if spacing is tight.

## Seeing it

Screenshot a single widget, themed:

```bash
poetry run python .claude/skills/pytweezer-gui-design/scripts/shoot.py \
    pytweezer.GUI.experiments.results:ResultsPanel /tmp/results.png --size 1500x950
```

Then Read the PNG. `--crop X,Y,W,H` zooms in on a region; `--wait` gives feeds
and polls time to settle. For a panel that needs a server, give it fakes. Write
a factory in a scratch directory and pass `--path`:

```python
# scratch/fakes.py: shoot.py fakes:panel out.png --path scratch
from PyQt6 import QtCore
from pytweezer.GUI.experiments.panel import ExperimentsPanel


class Feed(QtCore.QObject):  # same signals as the real feed
    queue_changed = QtCore.pyqtSignal(dict)
    point_received = QtCore.pyqtSignal(dict)
    connection_changed = QtCore.pyqtSignal(bool)

    def close(self):
        pass


def panel():
    p = ExperimentsPanel(client=FakeClient(), feed=Feed())
    QtCore.QTimer.singleShot(0, lambda: p.feed.queue_changed.emit(SNAPSHOT))
    return p
```

Use realistic content: several rows, a long label, a failed item, a running
item. Layouts that look fine empty break on real data. For the whole window,
use the `run-pytweezer` skill's driver; off the server PC it runs in
simulation.

After each screenshot, ask:
- Can I tell the regions apart, and which one I edit in?
- Is the primary action obvious?
- Is anything stretched, floating far from its label, truncated or cramped?
- Do disabled things look disabled?
- Could someone colour-blind read every state?
- Does every title, hint and empty state tell me what to do?

## Testing

Keep tests behavioural, offscreen (`qapp` fixture in `tests/conftest.py`). Assert
on what the user would see or trigger: a label's `state` property after an
edit, which buttons are enabled for a selection, that a refresh keeps the
selected item. Don't assert on pixels or colours. Construct panels with fake
clients and feeds, as `tests/test_experiment_gui.py` does.

## Worked example: the Experiments tab

**Before:** every region was `#1b1c22` or `#24252c`, about 5 % apart, with no
titles. Inputs were the same colour as the lists, values stretched across the
window, Submit was one grey button among many, and ten queue buttons were
always "enabled"-looking.

**After** (`pytweezer/GUI/experiments/`):
- The experiment list and the queue are wells; the editor is the sheet.
- Argument rows read label | value + unit | Scan, at fixed widths.
- Scan and Queue settings sit side by side.
- Submit is primary, with the point count beside it.
- Queue actions are grouped by target, with Abort and Delete in red at the
  right.
- The running row is tinted and finished rows recede.
- The queue title summarises state, and the Results tab updates itself instead
  of needing Refresh.
