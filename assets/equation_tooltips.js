// Attach hovercards to native KaTeX terms without changing their math layout.
const normalize = text => text.replace(/[\s\u200b-\u200d\ufeff]/g, "");
const selector = [
  ".st-key-home_moisture_path",
  ".st-key-home_thermal_path",
  ".st-key-home_combined_index",
  ".st-key-home_uptake_model"
].map(scope => `${scope} .katex-html .mord`).join(",");

// Dispose of handlers and any open card when Streamlit renders this again.
const previous = window.pcbEquationHints;
if (previous?.dispose) previous.dispose();
else {
  previous?.observer.disconnect();
  if (previous?.frame) cancelAnimationFrame(previous.frame);
}

const card = document.createElement("aside");
card.id = "pcb-equation-hovercard";
card.className = "pcb-equation-card";
card.setAttribute("role", "tooltip");
card.hidden = true;
document.body.appendChild(card);

const hints = {observer: null, rowObserver: null, frame: null};
const listeners = [];
const observedRows = new Set();
let activeTerm = null;
let showTimer = null;
let hideTimer = null;

function hideCard() {
  clearTimeout(showTimer);
  clearTimeout(hideTimer);
  activeTerm?.removeAttribute("aria-describedby");
  activeTerm = null;
  card.hidden = true;
}

function showCard(term) {
  clearTimeout(showTimer);
  clearTimeout(hideTimer);
  if (!term.isConnected) return;
  const definition = meanings[normalize(term.textContent)];
  if (!definition) return;
  activeTerm?.removeAttribute("aria-describedby");
  activeTerm = term;
  term.setAttribute("aria-describedby", card.id);

  const eyebrow = document.createElement("div");
  eyebrow.className = "pcb-equation-card__eyebrow";
  eyebrow.textContent = "Equation symbol";
  const symbol = document.createElement("div");
  symbol.className = "pcb-equation-card__symbol katex";
  const math = term.cloneNode(true);
  // The preview is decorative; keep the original term as the hover target.
  math.classList.remove("equation-symbol-hint");
  math.removeAttribute("tabindex");
  math.removeAttribute("role");
  math.removeAttribute("aria-label");
  math.removeAttribute("aria-describedby");
  math.setAttribute("aria-hidden", "true");
  symbol.appendChild(math);
  const description = document.createElement("p");
  description.className = "pcb-equation-card__meaning";
  description.textContent = definition.meaning;
  const units = document.createElement("div");
  units.className = "pcb-equation-card__units";
  const unitsLabel = document.createElement("span");
  unitsLabel.className = "pcb-equation-card__units-label";
  unitsLabel.textContent = "Units";
  const unitsValue = document.createElement("span");
  unitsValue.className = "pcb-equation-card__units-value";
  unitsValue.textContent = definition.units;
  units.append(unitsLabel, unitsValue);
  card.replaceChildren(eyebrow, symbol, description, units);

  // Render outside the equation containers so their overflow cannot crop it.
  card.hidden = false;
  card.style.visibility = "hidden";
  const target = term.getBoundingClientRect();
  const bounds = card.getBoundingClientRect();
  const viewportWidth = document.documentElement.clientWidth;
  const viewportHeight = window.innerHeight;
  const margin = 12;
  const clamp = (value, min, max) => Math.max(min, Math.min(value, max));
  const left = clamp(target.left + target.width / 2 - bounds.width / 2,
                     margin, viewportWidth - bounds.width - margin);
  let top = target.bottom + margin;
  if (top + bounds.height > viewportHeight - margin) {
    top = target.top - bounds.height - margin;
  }
  top = clamp(top, margin, viewportHeight - bounds.height - margin);
  card.style.left = `${left}px`;
  card.style.top = `${top}px`;
  card.style.visibility = "visible";
}

function scheduleHide() {
  clearTimeout(showTimer);
  clearTimeout(hideTimer);
  // Leave time to move the pointer from a symbol into its card.
  hideTimer = setTimeout(hideCard, 180);
}

function termFor(target) {
  return target instanceof Element ? target.closest(".equation-symbol-hint") : null;
}

function listen(target, name, handler, capture = false) {
  target.addEventListener(name, handler, capture);
  listeners.push(() => target.removeEventListener(name, handler, capture));
}

listen(document, "pointerover", event => {
  const term = termFor(event.target);
  if (!term || term === termFor(event.relatedTarget)) return;
  clearTimeout(hideTimer);
  clearTimeout(showTimer);
  showTimer = setTimeout(() => showCard(term), 160);
});
listen(document, "pointerout", event => {
  const term = termFor(event.target);
  if (!term || term === termFor(event.relatedTarget)) return;
  if (!card.contains(event.relatedTarget)) scheduleHide();
});
listen(card, "pointerenter", () => clearTimeout(hideTimer));
listen(card, "pointerleave", scheduleHide);
listen(document, "focusin", event => {
  const term = termFor(event.target);
  if (term) showCard(term);
});
listen(document, "focusout", event => {
  if (termFor(event.target)) scheduleHide();
});
listen(document, "click", event => {
  const term = termFor(event.target);
  if (term) showCard(term);
  else if (!card.contains(event.target)) hideCard();
});
listen(document, "keydown", event => {
  if (event.key === "Escape") hideCard();
  const term = termFor(event.target);
  if (term && (event.key === "Enter" || event.key === " ")) {
    event.preventDefault();
    showCard(term);
  }
});
listen(window, "scroll", event => {
  if (event.target instanceof Element && card.contains(event.target)) return;
  hideCard();
}, true);
listen(window, "resize", hideCard);

function alignFlowRows() {
  observedRows.forEach(row => {
    if (!row.isConnected) {
      hints.rowObserver.unobserve(row);
      observedRows.delete(row);
    }
  });
  for (const row of ["heading", "step_1", "step_2", "step_3"]) {
    const left = document.querySelector(`.st-key-home_moisture_${row}`);
    const right = document.querySelector(`.st-key-home_thermal_${row}`);
    if (!left || !right) continue;
    for (const element of [left, right]) {
      // Measure natural content again when equations, fonts or widths change.
      element.style.minHeight = "";
      if (!observedRows.has(element)) {
        hints.rowObserver.observe(element);
        observedRows.add(element);
      }
    }
    const leftBounds = left.getBoundingClientRect();
    const rightBounds = right.getBoundingClientRect();
    // A stacked mobile layout does not need extra space between steps.
    if (Math.abs(leftBounds.left - rightBounds.left) < 1) continue;
    const height = `${Math.ceil(Math.max(leftBounds.height, rightBounds.height))}px`;
    left.style.minHeight = height;
    right.style.minHeight = height;
  }
}

function attachHints() {
  hints.frame = null;
  alignFlowRows();
  if (activeTerm && !activeTerm.isConnected) hideCard();
  // Annotate parents first so a composite term has one consistent definition.
  document.querySelectorAll(selector).forEach(term => {
    if (term.parentElement.closest(".equation-symbol-hint")) return;
    const definition = meanings[normalize(term.textContent)];
    if (!definition) return;
    term.classList.add("equation-symbol-hint");
    term.removeAttribute("title");
    term.setAttribute("tabindex", "0");
    term.setAttribute("role", "button");
    term.setAttribute("aria-label", `${definition.meaning}. Units: ${definition.units}`);
  });
}

function scheduleUpdate() {
  if (hints.frame === null) hints.frame = requestAnimationFrame(attachHints);
}
hints.rowObserver = new ResizeObserver(scheduleUpdate);
hints.observer = new MutationObserver(scheduleUpdate);
hints.observer.observe(document.querySelector('[data-testid="stMain"]') || document.body,
                       {childList: true, subtree: true});
hints.dispose = () => {
  hideCard();
  hints.observer.disconnect();
  hints.rowObserver.disconnect();
  if (hints.frame !== null) cancelAnimationFrame(hints.frame);
  listeners.forEach(remove => remove());
  card.remove();
};
window.pcbEquationHints = hints;
attachHints();
