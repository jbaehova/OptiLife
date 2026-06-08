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
  ratingWeight: 1,
  workloadWeight: 1.1,
  teamworkWeight: 0.8,
  gradingWeight: 0.9,
};

const state = {
  recommendations: [],
  selectedIndex: 0,
  healthStatusText: "",
};

const els = {
  form: document.querySelector("#conditionForm"),
  resetButton: document.querySelector("#resetButton"),
  syncReviewsButton: document.querySelector("#syncReviewsButton"),
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
  ratingWeight: document.querySelector("#ratingWeight"),
  workloadWeight: document.querySelector("#workloadWeight"),
  teamworkWeight: document.querySelector("#teamworkWeight"),
  gradingWeight: document.querySelector("#gradingWeight"),
  coursesDatasetCount: document.querySelector("#coursesDatasetCount"),
  coursesDatasetPath: document.querySelector("#coursesDatasetPath"),
  rawReviewsDatasetCount: document.querySelector("#rawReviewsDatasetCount"),
  rawReviewsDatasetPath: document.querySelector("#rawReviewsDatasetPath"),
  labeledReviewsDatasetCount: document.querySelector("#labeledReviewsDatasetCount"),
  labeledReviewsDatasetPath: document.querySelector("#labeledReviewsDatasetPath"),
  reviewFloorStatus: document.querySelector("#reviewFloorStatus"),
  reviewFloorDetail: document.querySelector("#reviewFloorDetail"),
  syncMessage: document.querySelector("#syncMessage"),
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

function setSyncMessage(text, type = "") {
  els.syncMessage.textContent = text;
  els.syncMessage.className = `sync-message ${type}`.trim();
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
  renderDatasetStatus(health.dataset);
  state.healthStatusText =
    health.dataset.ready ? "CSV 준비됨" : "데이터 확인 필요";
  setStatus(state.healthStatusText, "ready");
}

function datasetCountLabel(dataset, unit) {
  if (!dataset.exists) {
    return "없음";
  }
  return `${dataset.row_count}${unit}`;
}

function renderDatasetStatus(dataset) {
  els.coursesDatasetCount.textContent = datasetCountLabel(dataset.courses, "개");
  els.coursesDatasetPath.textContent = dataset.courses.path;
  els.rawReviewsDatasetCount.textContent = datasetCountLabel(
    dataset.raw_reviews,
    "개",
  );
  els.rawReviewsDatasetPath.textContent = dataset.raw_reviews.path;
  els.labeledReviewsDatasetCount.textContent = datasetCountLabel(
    dataset.labeled_reviews,
    "개",
  );
  els.labeledReviewsDatasetPath.textContent = dataset.labeled_reviews.path;

  const missing = dataset.courses_missing_evaluation || [];
  const missingCount = missing.length;
  els.reviewFloorStatus.textContent =
    missingCount === 0 ? "충족" : `${missingCount}개 누락`;
  els.reviewFloorDetail.textContent =
    missingCount === 0
      ? "에브리타임 평가 컬럼 준비됨"
      : missing.join(", ");
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

async function syncReviews() {
  els.syncReviewsButton.disabled = true;
  setSyncMessage("리뷰 라벨링 실행 중");
  try {
    const data = await getJson("/api/admin/sync-reviews", { method: "POST" });
    renderDatasetStatus(data.dataset);
    setSyncMessage(data.message || data.stdout || "리뷰 동기화 완료", "ready");
    state.healthStatusText = data.dataset.ready ? "CSV 준비됨" : "데이터 확인 필요";
    setStatus(state.healthStatusText, "ready");
  } catch (error) {
    setSyncMessage(error.message, "error");
  } finally {
    els.syncReviewsButton.disabled = false;
  }
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
  const lowestScore = Math.min(
    block.workload_label,
    block.teamwork_load_label,
    block.grading_strictness_label,
  );
  if (lowestScore <= 2) {
    return "level-high";
  }
  if (lowestScore <= 3) {
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
  const score = (value) => Number(value).toFixed(2);
  const courseItems = selected.courses
    .map((course) => {
      const core = course.core ? "전공필수" : "선택";
      return `
        <div class="summary-item">
          <strong>${escapeHtml(course.course_name)}</strong>
          <span>${course.credits}학점 · ${core} · 평점 ${score(course.rating)} · 과제 ${score(course.workload_label)} · 조모임 ${score(course.teamwork_load_label)} · 성적 ${score(course.grading_strictness_label)}</span>
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

els.syncReviewsButton.addEventListener("click", syncReviews);

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
