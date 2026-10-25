/** Save rows as a CSV file. Values are written raw (not display-formatted), so
 * the export is data, not a picture of the table. */
export function downloadCSV(filename: string, header: string[], rows: (string | number | boolean | null | undefined)[][]) {
  const escape = (value: string | number | boolean | null | undefined) => {
    if (value === null || value === undefined) return "";
    const text = String(value);
    return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  const body = [header, ...rows].map((row) => row.map(escape).join(",")).join("\n");
  const url = URL.createObjectURL(new Blob([body], { type: "text/csv;charset=utf-8" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
