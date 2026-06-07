const days = [
  ["Mon", "월"],
  ["Tue", "화"],
  ["Wed", "수"],
  ["Thu", "목"],
  ["Fri", "금"],
];

const defaults = {
  minCredits: 15,
  maxCredits: 18,
  limit: 3,
  freeDays: ["Fri"],
  avoidEarly: true,
  avoidFridayAfternoon: true,
  balanceDays: true,
  earlyCutoff: "10:00",
  coreWeight: 8,
  difficultyWeight: 1.3,
  workloadWeight: 1.1,
};

const state = {
  recommendations: [],
  selectedIndex: 0,
  healthStatusText: "",
};

const els = {
  form: document.querySelector("#conditionForm"),
  resetButton: document.querySelector("#resetButton"),
  resultsArea: document.querySelector("#resultsArea"),
  apiStatus: document.querySelector("#apiStatus"),
  resultTitle: document.querySelector("#resultTitle"),
  courseCount: document.querySelector("#courseCount"),
  dataSource: document.querySelector("#dataSource"),
  tabs: document.querySelector("#recommendationTabs"),
  calendar: document.querySelector("#calendar"),
  summary: document.querySelector("#summary"),
  payload: document.querySelector("#conditionPayload"),
  minCredits: document.querySelector("#minCredits"),
  maxCredits: document.querySelector("#maxCredits"),
  limit: document.querySelector("#limit"),
  freeDays: document.querySelector("#freeDays"),
  avoidEarly: document.querySelector("#avoidEarly"),
  avoidFridayAfternoon: document.querySelector("#avoidFridayAfternoon"),
  balanceDays: document.querySelector("#balanceDays"),
  earlyCutoff: document.querySelector("#earlyCutoff"),
  coreWeight: document.querySelector("#coreWeight"),
  difficultyWeight: document.querySelector("#difficultyWeight"),
  workloadWeight: document.querySelector("#workloadWeight"),
};

function toNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function buildCondition() {
  return {
    min_credits: toNumber(els.minCredits.value, defaults.minCredits),
    max_credits: toNumber(els.maxCredits.value, defaults.maxCredits),
    limit: toNumber(els.limit.value, defaults.limit),
    preferred_free_days: Array.from(
      els.freeDays.querySelectorAll("input:checked"),
    ).map((input) => input.value),
    avoid_early: els.avoidEarly.checked,
    avoid_friday_afternoon: els.avoidFridayAfternoon.checked,
    balance_days: els.balanceDays.checked,
    early_cutoff: els.earlyCutoff.value || defaults.earlyCutoff,
    core_weight: toNumber(els.coreWeight.value, defaults.coreWeight),
    difficulty_weight: toNumber(
      els.difficultyWeight.value,
      defaults.difficultyWeight,
    ),
    workload_weight: toNumber(els.workloadWeight.value, defaults.workloadWeight),
  };
}

function updatePayload() {
  els.payload.textContent = JSON.stringify(buildCondition(), null, 2);
}

function updateRangeOutputs() {
  document.querySelectorAll("[data-output-for]").forEach((output) => {
    const input = document.querySelector(`#${output.dataset.outputFor}`);
    output.textContent = input.value;
  });
}

function setStatus(text, type = "") {
  els.apiStatus.textContent = text;
  els.apiStatus.className = `status ${type}`.trim();
}

function showResults() {
  els.resultsArea.hidden = false;
}

function hideResults() {
  els.resultsArea.hidden = true;
  state.recommendations = [];
  state.selectedIndex = 0;
  els.resultTitle.textContent = "시간표를 계산 중입니다";
  els.tabs.innerHTML = "";
  els.calendar.innerHTML = "";
  els.summary.innerHTML = "";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function getJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function loadHealth() {
  const health = await getJson("/api/health");
  els.courseCount.textContent = health.course_count;
  els.dataSource.textContent = health.data_source;
  state.healthStatusText =
    health.data_status === "sample" ? "샘플 CSV 사용 중" : "CSV 준비됨";
  setStatus(state.healthStatusText, "ready");
}

async function requestRecommendations() {
  const condition = buildCondition();
  showResults();
  updatePayload();
  els.resultTitle.textContent = "시간표를 계산 중입니다";
  els.tabs.innerHTML = "";
  els.summary.innerHTML = "";

  const data = await getJson("/api/recommend", {
    method: "POST",
    body: JSON.stringify(condition),
  });

  els.courseCount.textContent = data.course_count;
  els.dataSource.textContent = data.data_source;
  state.recommendations = data.recommendations;
  state.selectedIndex = 0;
  renderResults();
}

function renderResults() {
  if (!state.recommendations.length) {
    els.resultTitle.textContent = "조건에 맞는 시간표가 없습니다";
    els.tabs.innerHTML = "";
    els.calendar.innerHTML = `<div class="summary-empty">학점 범위나 공강 조건을 조금 낮춰야 합니다.</div>`;
    els.summary.innerHTML = `<div class="summary-empty">추천 후보 0개</div>`;
    return;
  }

  renderTabs();
  renderSelected();
}

function renderTabs() {
  els.tabs.innerHTML = state.recommendations
    .map((item, index) => {
      const active = index === state.selectedIndex ? " active" : "";
      return `<button class="tab${active}" type="button" data-index="${index}">#${item.rank} 점수 ${item.score}</button>`;
    })
    .join("");
}

function renderSelected() {
  const selected = state.recommendations[state.selectedIndex];
  els.resultTitle.textContent = `${selected.credits}학점 추천 #${selected.rank}`;
  renderCalendar(selected.blocks);
  renderSummary(selected);
}

function minutesToPercent(minutes) {
  const start = 8 * 60;
  const end = 19 * 60;
  return ((minutes - start) / (end - start)) * 100;
}

function eventLevel(block) {
  if (block.difficulty_label >= 4 || block.workload_label >= 4) {
    return "level-high";
  }
  if (block.difficulty_label === 3 || block.workload_label === 3) {
    return "level-mid";
  }
  return "level-low";
}

function renderCalendar(blocks) {
  const hourMarks = Array.from({ length: 12 }, (_, index) => 8 + index);
  const blocksByDay = Object.groupBy
    ? Object.groupBy(blocks, (block) => block.day)
    : blocks.reduce((grouped, block) => {
        grouped[block.day] = grouped[block.day] || [];
        grouped[block.day].push(block);
        return grouped;
      }, {});

  const headers = days
    .map(([, label]) => `<div class="day-head">${label}</div>`)
    .join("");
  const timeAxis = `
    <div class="time-axis">
      ${hourMarks
        .map((hour) => {
          const top = minutesToPercent(hour * 60);
          return `<span class="time-mark" style="top:${top}%">${hour}:00</span>`;
        })
        .join("")}
    </div>`;
  const lanes = days
    .map(([day]) => {
      const events = (blocksByDay[day] || [])
        .map((block) => {
          const top = Math.max(0, minutesToPercent(block.start_minutes));
          const height = Math.max(
            7,
            minutesToPercent(block.end_minutes) - minutesToPercent(block.start_minutes),
          );
          return `
            <div class="event ${eventLevel(block)}" style="top:${top}%;height:${height}%">
              <strong>${escapeHtml(block.course_name)}</strong>
              <span>${block.start}-${block.end}</span>
            </div>`;
        })
        .join("");
      return `<div class="day-lane">${events}</div>`;
    })
    .join("");

  els.calendar.innerHTML = `
    <div class="calendar-grid">
      <div class="time-head"></div>
      ${headers}
      ${timeAxis}
      ${lanes}
    </div>`;
}

function renderSummary(selected) {
  const courseItems = selected.courses
    .map((course) => {
      const core = course.core ? "전공필수" : "선택";
      return `
        <div class="summary-item">
          <strong>${escapeHtml(course.course_name)}</strong>
          <span>${course.credits}학점 · ${core} · 난이도 ${course.difficulty_label} · 과제량 ${course.workload_label}</span>
        </div>`;
    })
    .join("");
  const reasonItems = selected.reasons
    .map((reason) => `<div class="summary-item">${escapeHtml(reason)}</div>`)
    .join("");
  const burdenItems = Object.entries(selected.daily_burden)
    .map(
      ([day, value]) =>
        `<div class="summary-item"><strong>${escapeHtml(day)}</strong><span>부담 ${value}</span></div>`,
    )
    .join("");

  els.summary.innerHTML = `
    <div class="summary-group">
      <h2>과목</h2>
      ${courseItems}
    </div>
    <div class="summary-group">
      <h2>추천 이유</h2>
      ${reasonItems}
    </div>
    <div class="summary-group">
      <h2>요일 부담</h2>
      ${burdenItems || `<div class="summary-item">-</div>`}
    </div>`;
}

function resetForm() {
  els.minCredits.value = defaults.minCredits;
  els.maxCredits.value = defaults.maxCredits;
  els.limit.value = defaults.limit;
  els.freeDays.querySelectorAll("input").forEach((input) => {
    input.checked = defaults.freeDays.includes(input.value);
  });
  els.avoidEarly.checked = defaults.avoidEarly;
  els.avoidFridayAfternoon.checked = defaults.avoidFridayAfternoon;
  els.balanceDays.checked = defaults.balanceDays;
  els.earlyCutoff.value = defaults.earlyCutoff;
  els.coreWeight.value = defaults.coreWeight;
  els.difficultyWeight.value = defaults.difficultyWeight;
  els.workloadWeight.value = defaults.workloadWeight;
  updateRangeOutputs();
  updatePayload();
}

els.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await requestRecommendations();
    setStatus("추천 완료", "ready");
  } catch (error) {
    setStatus("API 오류", "error");
    els.resultTitle.textContent = "추천 계산 실패";
    els.calendar.innerHTML = `<div class="error-box">${escapeHtml(error.message)}</div>`;
  }
});

els.resetButton.addEventListener("click", async () => {
  resetForm();
  hideResults();
  setStatus(state.healthStatusText || "조건 입력 대기", "ready");
});

els.tabs.addEventListener("click", (event) => {
  const button = event.target.closest("[data-index]");
  if (!button) {
    return;
  }
  state.selectedIndex = Number(button.dataset.index);
  renderTabs();
  renderSelected();
});

els.form.addEventListener("input", () => {
  updateRangeOutputs();
  updatePayload();
});

resetForm();
hideResults();
loadHealth()
  .catch((error) => {
    setStatus("API 연결 실패", "error");
  });
