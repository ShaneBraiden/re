/* The only module that talks to the server. Responses are cached per URL for
 * the session - a finished run does not change under the page - except the
 * listings that a new run or a new upload can grow. */

const cache = new Map();

export async function getJSON(url, { fresh = false } = {}) {
  if (!fresh && cache.has(url)) return cache.get(url);
  const request = fetch(url).then(async (response) => {
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error || body.note || `${response.status} ${response.statusText}`);
    return body;
  });
  if (!fresh) {
    cache.set(url, request);
    request.catch(() => cache.delete(url));
  }
  return request;
}

export function query(params) {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined || value === "") continue;
    search.set(key, Array.isArray(value) ? value.join(",") : String(value));
  }
  return search.toString();
}

export const api = {
  async health() {
    // 503 is an answer here ("no run yet"), not a failure.
    const response = await fetch("/api/health");
    return response.json().catch(() => ({ ready: false, upload: false }));
  },
  experiments: () => getJSON("/api/experiments", { fresh: true }),
  index: (exp) => getJSON(`/api/index?${query({ exp })}`),
  values: (exp, image, condition, pipelines) =>
    getJSON(`/api/values?${query({ exp, image, condition, pipelines })}`),
  series: (exp, image, metric, pipelines) =>
    getJSON(`/api/series?${query({ exp, image, metric, pipelines })}`),
  aggregate: (exp, metric, pipelines) =>
    getJSON(`/api/aggregate?${query({ exp, metric, pipelines })}`),
  run: (exp) => getJSON(`/api/run?${query({ exp })}`),
  analysis: (exp) => getJSON(`/api/analysis?${query({ exp })}`),
  enhancements: () => getJSON("/api/enhancements", { fresh: true }),
};

export function previewURL(exp, image, which, conditionHash) {
  const name = which === "__reference__" ? "reference.jpg"
    : which === "__degraded__" ? `${conditionHash}__degraded.jpg`
      : `${conditionHash}__${which}.jpg`;
  return `/previews/${encodeURIComponent(image)}/${name}?${query({ exp })}`;
}
