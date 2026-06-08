const days = [
  ["Mon", "월"],
  ["Tue", "화"],
  ["Wed", "수"],
  ["Thu", "목"],
  ["Fri", "금"],
];

const categoryOrder = ["전공코어", "전공심화", "실험실습", "교양", "전공(대학원)"];

const defaults = {
  minCredits: 15,
  maxCredits: 18,
  coreMinCount: 2,
  coreMaxCount: 4,
  limit: 3,
  freeDays: ["Fri"],
  avoidEarly: true,
  avoidFridayAfternoon: true,
  balanceDays: true,
  earlyCutoff: "10:00",
  ratingWeight: 1,
  workloadWeight: 1.1,
  teamworkWeight: 0.8,
  gradingWeight: 0.9,
};

const state = {
  recommendations: [],
  diagnostics: { status: "idle", blocking: [], warnings: [] },
  selectedIndex: 0,
  healthStatusText: "",
  categoryCounts: {},
};

const els = {
  form: document.querySelector("#conditionForm"),
  resetButton: document.querySelector("#resetButton"),
  resultsArea: document.querySelector("#resultsArea"),
  apiStatus: document.querySelector("#apiStatus"),
  resultTitle: document.querySelector("#resultTitle"),
  courseCount: document.querySelector("#courseCount"),
  dataSource: document.querySelector("#dataSource"),
  selectedCredits: document.querySelector("#selectedCredits"),
  selectedCoreCount: document.querySelector("#selectedCoreCount"),
  selectedScore: document.querySelector("#selectedScore"),
  tabs: document.querySelector("#recommendationTabs"),
  calendar: document.querySelector("#calendar"),
  summary: document.querySelector("#summary"),
  diagnostics: document.querySelector("#diagnostics"),
  payload: document.querySelector("#conditionPayload"),
  minCredits: document.querySelector("#minCredits"),
  maxCredits: document.querySelector("#maxCredits"),
  coreMinCount: document.querySelector("#coreMinCount"),
  coreMaxCount: document.querySelector("#coreMaxCount"),
  limit: document.querySelector("#limit"),
  categoryOptions: document.querySelector("#categoryOptions"),
  freeDays: document.querySelector("#freeDays"),
  avoidEarly: document.querySelector("#avoidEarly"),
  avoidFridayAfternoon: document.querySelector("#avoidFridayAfternoon"),
  balanceDays: document.querySelector("#balanceDays"),
  earlyCutoff: document.querySelector("#earlyCutoff"),
  ratingWeight: document.querySelector("#ratingWeight"),
  workloadWeight: document.querySelector("#workloadWeight"),
  teamworkWeight: document.querySelector("#teamworkWeight"),
  gradingWeight: document.querySelector("#gradingWeight"),
  coursesDatasetCount: document.querySelector("#coursesDatasetCount"),
  coreDatasetCount: document.querySelector("#coreDatasetCount"),
  categoryDatasetCount: document.querySelector("#categoryDatasetCount"),
  reviewFloorStatus: document.querySelector("#reviewFloorStatus"),
};

function toNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function checkedValues(container) {
  return Array.from(container.querySelectorAll("input:checked")).map(
    (input) => input.value,
  );
}

function buildCondition() {
  return {
    min_credits: toNumber(els.minCredits.value, defaults.minCredits),
    max_credits: toNumber(els.maxCredits.value, defaults.maxCredits),
    core_min_count: toNumber(els.coreMinCount.value, defaults.coreMinCount),
    core_max_count: toNumber(els.coreMaxCount.value, defaults.coreMaxCount),
    limit: toNumber(els.limit.value, defaults.limit),
    categories: checkedValues(els.categoryOptions),
    preferred_free_days: checkedValues(els.freeDays),
    avoid_early: els.avoidEarly.checked,
    avoid_friday_afternoon: els.avoidFridayAfternoon.checked,
    balance_days: els.balanceDays.checked,
    early_cutoff: els.earlyCutoff.value || defaults.earlyCutoff,
    rating_weight: toNumber(els.ratingWeight.value, defaults.ratingWeight),
    workload_weight: toNumber(els.workloadWeight.value, defaults.workloadWeight),
    teamwork_weight: toNumber(els.teamworkWeight.value, defaults.teamworkWeight),
    grading_weight: toNumber(els.gradingWeight.value, defaults.gradingWeight),
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
  els.resultsArea.hidden = false;
  state.recommendations = [];
  state.diagnostics = { status: "idle", blocking: [], warnings: [] };
  state.selectedIndex = 0;
  els.resultTitle.textContent = "추천 대기";
  els.selectedCredits.textContent = "-";
  els.selectedCoreCount.textContent = "-";
  els.selectedScore.textContent = "-";
  els.tabs.innerHTML = "";
  els.calendar.innerHTML = `
    <div class="empty-state">
      <strong>조건을 조정한 뒤 추천 계산을 실행하세요.</strong>
      <span>전공필수 개수 범위와 카테고리 필터를 기준으로 courses.csv에서 시간표를 만듭니다.</span>
    </div>`;
  els.summary.innerHTML = `<div class="summary-empty">선택된 추천 없음</div>`;
  els.diagnostics.innerHTML = `<div class="summary-empty">진단 없음</div>`;
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
  els.courseCount.textContent = `${health.course_count}개 과목`;
  els.dataSource.textContent = health.data_source;
  renderDatasetStatus(health.dataset);
  renderCategoryOptions(health.dataset.categories || {});
  state.healthStatusText = health.dataset.ready ? "CSV 준비됨" : "데이터 확인 필요";
  setStatus(state.healthStatusText, health.dataset.ready ? "ready" : "error");
}

function orderedCategoryEntries(categoryCounts) {
  const entries = Object.entries(categoryCounts);
  return entries.sort(([left], [right]) => {
    const leftIndex = categoryOrder.indexOf(left);
    const rightIndex = categoryOrder.indexOf(right);
    if (leftIndex !== -1 || rightIndex !== -1) {
      return (leftIndex === -1 ? 99 : leftIndex) - (rightIndex === -1 ? 99 : rightIndex);
    }
    return left.localeCompare(right, "ko");
  });
}

function renderCategoryOptions(categoryCounts) {
  state.categoryCounts = categoryCounts;
  const entries = orderedCategoryEntries(categoryCounts);
  if (!entries.length) {
    els.categoryOptions.innerHTML = `<div class="empty-inline">카테고리 없음</div>`;
    updatePayload();
    return;
  }

  els.categoryOptions.innerHTML = entries
    .map(
      ([category, count]) => `
        <label>
          <input type="checkbox" value="${escapeHtml(category)}" checked>
          <span>${escapeHtml(category)}</span>
          <small>${count}</small>
        </label>`,
    )
    .join("");
  updatePayload();
}

function renderDatasetStatus(dataset) {
  const missing = dataset.courses_missing_evaluation || [];
  els.coursesDatasetCount.textContent = dataset.courses.exists
    ? `${dataset.courses.row_count}개`
    : "없음";
  els.coreDatasetCount.textContent = `${dataset.core_course_count || 0}개`;
  els.categoryDatasetCount.textContent = `${Object.keys(dataset.categories || {}).length}개`;
  els.reviewFloorStatus.textContent = missing.length === 0 ? "충족" : `${missing.length}개 누락`;
}

async function requestRecommendations() {
  const condition = buildCondition();
  showResults();
  updatePayload();
  els.resultTitle.textContent = "시간표를 계산 중입니다";
  els.tabs.innerHTML = "";
  els.summary.innerHTML = "";
  els.diagnostics.innerHTML = "";

  const data = await getJson("/api/recommend", {
    method: "POST",
    body: JSON.stringify(condition),
  });

  els.courseCount.textContent = `${data.course_count}개 과목`;
  els.dataSource.textContent = data.data_source;
  state.recommendations = data.recommendations;
  state.diagnostics = data.diagnostics;
  state.selectedIndex = 0;
  renderResults();
}

function renderResults() {
  if (!state.recommendations.length) {
    els.resultTitle.textContent = "조건에 맞는 시간표가 없습니다";
    els.selectedCredits.textContent = "-";
    els.selectedCoreCount.textContent = "-";
    els.selectedScore.textContent = "-";
    els.tabs.innerHTML = "";
    els.calendar.innerHTML = renderDiagnosticEmpty(state.diagnostics.blocking);
    els.summary.innerHTML = `<div class="summary-empty">추천 후보 0개</div>`;
    renderDiagnostics(state.diagnostics.blocking, "조정 도움말");
    return;
  }

  renderTabs();
  renderSelected();
}

function renderTabs() {
  els.tabs.innerHTML = state.recommendations
    .map((item, index) => {
      const active = index === state.selectedIndex ? " active" : "";
      return `<button class="tab${active}" type="button" data-index="${index}">#${item.rank} · ${item.credits}학점 · 필수 ${item.core_count}</button>`;
    })
    .join("");
}

function renderSelected() {
  const selected = state.recommendations[state.selectedIndex];
  els.resultTitle.textContent = `${selected.credits}학점 추천 #${selected.rank}`;
  els.selectedCredits.textContent = `${selected.credits}`;
  els.selectedCoreCount.textContent = `${selected.core_count}`;
  els.selectedScore.textContent = `${selected.score}`;
  renderCalendar(selected.blocks);
  renderSummary(selected);
  renderDiagnostics(selected.unmet_preferences, "미충족 선호");
}

function minutesToPercent(minutes) {
  const start = 8 * 60;
  const end = 19 * 60;
  return ((minutes - start) / (end - start)) * 100;
}

function eventClass(block) {
  return block.core ? "event-core" : "event-elective";
}

function groupBlocksByDay(blocks) {
  return blocks.reduce((grouped, block) => {
    grouped[block.day] = grouped[block.day] || [];
    grouped[block.day].push(block);
    return grouped;
  }, {});
}

function renderCalendar(blocks) {
  const hourMarks = Array.from({ length: 12 }, (_, index) => 8 + index);
  const blocksByDay = groupBlocksByDay(blocks);
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
          const courseType = block.core ? "전공필수" : `${block.credits}학점`;
          return `
            <div class="event ${eventClass(block)}" style="top:${top}%;height:${height}%">
              <strong>${escapeHtml(block.course_name)}</strong>
              <span>${escapeHtml(block.start)}-${escapeHtml(block.end)} · ${escapeHtml(courseType)}</span>
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

function renderDiagnosticEmpty(items) {
  const content = (items || [])
    .map(
      (item) => `
        <div class="diagnostic-item">
          <strong>${escapeHtml(item.label)}</strong>
          <span>${escapeHtml(item.actual)}</span>
          <p>${escapeHtml(item.help)}</p>
        </div>`,
    )
    .join("");
  return `<div class="empty-state">${content || "조건을 조정해야 합니다."}</div>`;
}

function score(value) {
  return Number(value).toFixed(2);
}

function renderSummary(selected) {
  const courseItems = selected.courses
    .map((course) => {
      const core = course.core ? "전공필수" : "선택";
      const professor = course.professor || "교수 미정";
      const section = course.section ? `${course.section}분반` : "분반 미정";
      const time = course.time_slot || "시간 미지정";
      return `
        <div class="summary-item">
          <strong>${escapeHtml(course.course_name)}</strong>
          <span>${course.credits}학점 · ${core} · ${escapeHtml(course.category || "미분류")} · ${escapeHtml(professor)} · ${escapeHtml(section)}</span>
          <span>${escapeHtml(time)}</span>
          <span>평점 ${score(course.rating)} · 과제 ${score(course.workload_label)} · 조모임 ${score(course.teamwork_load_label)} · 성적 ${score(course.grading_strictness_label)}</span>
        </div>`;
    })
    .join("");
  const reasonItems = selected.reasons
    .map((reason) => `<div class="summary-item">${escapeHtml(reason)}</div>`)
    .join("");
  const categoryItems = Object.entries(selected.category_counts)
    .map(
      ([category, count]) =>
        `<div class="summary-item"><strong>${escapeHtml(category)}</strong><span>${count}개</span></div>`,
    )
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
      <h2>카테고리</h2>
      ${categoryItems || `<div class="summary-item">-</div>`}
    </div>
    <div class="summary-group">
      <h2>요일 부담</h2>
      ${burdenItems || `<div class="summary-item">-</div>`}
    </div>`;
}

function renderDiagnostics(items, title) {
  const list = items || [];
  if (!list.length) {
    els.diagnostics.innerHTML = `<div class="summary-empty">${escapeHtml(title)} 없음</div>`;
    return;
  }

  els.diagnostics.innerHTML = list
    .map(
      (item) => `
        <div class="diagnostic-item">
          <strong>${escapeHtml(item.label)}</strong>
          <span>${escapeHtml(item.expected)} → ${escapeHtml(item.actual)}</span>
          <p>${escapeHtml(item.help)}</p>
        </div>`,
    )
    .join("");
}

function resetForm() {
  els.minCredits.value = defaults.minCredits;
  els.maxCredits.value = defaults.maxCredits;
  els.coreMinCount.value = defaults.coreMinCount;
  els.coreMaxCount.value = defaults.coreMaxCount;
  els.limit.value = defaults.limit;
  els.categoryOptions.querySelectorAll("input").forEach((input) => {
    input.checked = true;
  });
  els.freeDays.querySelectorAll("input").forEach((input) => {
    input.checked = defaults.freeDays.includes(input.value);
  });
  els.avoidEarly.checked = defaults.avoidEarly;
  els.avoidFridayAfternoon.checked = defaults.avoidFridayAfternoon;
  els.balanceDays.checked = defaults.balanceDays;
  els.earlyCutoff.value = defaults.earlyCutoff;
  els.ratingWeight.value = defaults.ratingWeight;
  els.workloadWeight.value = defaults.workloadWeight;
  els.teamworkWeight.value = defaults.teamworkWeight;
  els.gradingWeight.value = defaults.gradingWeight;
  updateRangeOutputs();
  updatePayload();
}

els.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await requestRecommendations();
    setStatus("추천 완료", "ready");
  } catch (error) {
    showResults();
    setStatus("API 오류", "error");
    els.resultTitle.textContent = "추천 계산 실패";
    els.calendar.innerHTML = `<div class="error-box">${escapeHtml(error.message)}</div>`;
  }
});

els.resetButton.addEventListener("click", () => {
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
loadHealth().catch(() => {
  setStatus("API 연결 실패", "error");
});
