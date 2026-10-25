import { useCallback, useState } from "react";

export type Theme = "auto" | "light" | "dark";
const KEY = "sensorsim-theme";

function stored(): Theme {
  try {
    const value = localStorage.getItem(KEY);
    return value === "light" || value === "dark" ? value : "auto";
  } catch {
    return "auto";
  }
}

/** Auto follows the OS; light and dark pin it. index.html applies the stored
 * choice before first paint so it never flashes. */
export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(stored);
  const setTheme = useCallback((next: Theme) => {
    const root = document.documentElement;
    if (next === "auto") delete root.dataset.theme;
    else root.dataset.theme = next;
    try {
      localStorage.setItem(KEY, next);
    } catch {
      /* storage blocked: the choice lasts for this page */
    }
    setThemeState(next);
  }, []);
  return [theme, setTheme] as const;
}
