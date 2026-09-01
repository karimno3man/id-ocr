/**
 * Google Apps Script for iSchool ID-OCR submissions.
 *
 * Setup:
 * 1. Create a Google Sheet with row 1 headers (exact names):
 *    First_Name, Last_Name, HusbandName, Gender, Religion, Status, ID,
 *    IssueDate, ExpDate, Serial_Num, Add1, Add2, Job1, Job2, Front, Back,
 *    submitted_at
 * 2. Extensions → Apps Script → paste this file → Save
 * 3. Set WEBHOOK_TOKEN below (optional) to match GOOGLE_SHEETS_TOKEN on the server
 * 4. Deploy → New deployment → Web app
 *    Execute as: Me
 *    Who has access: Anyone
 * 5. Copy the Web app URL ending in /exec into GOOGLE_SHEETS_WEBHOOK_URL
 */

const FIELD_COLUMNS = [
  "First_Name",
  "Last_Name",
  "HusbandName",
  "Gender",
  "Religion",
  "Status",
  "ID",
  "IssueDate",
  "ExpDate",
  "Serial_Num",
  "Add1",
  "Add2",
  "Job1",
  "Job2",
  "Front",
  "Back",
];

/** Must match GOOGLE_SHEETS_TOKEN on the FastAPI server (leave empty to disable). */
const WEBHOOK_TOKEN = "";

function doPost(e) {
  try {
    if (!e || !e.postData || !e.postData.contents) {
      return jsonResponse({ ok: false, error: "Missing request body" });
    }

    const body = JSON.parse(e.postData.contents);
    if (WEBHOOK_TOKEN && body.token !== WEBHOOK_TOKEN) {
      return jsonResponse({ ok: false, error: "Unauthorized" });
    }

    const fields = body.fields || {};
    const row = FIELD_COLUMNS.map((name) => String(fields[name] ?? ""));
    row.push(body.submitted_at || new Date().toISOString());

    const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
    sheet.appendRow(row);

    return jsonResponse({ ok: true });
  } catch (err) {
    return jsonResponse({ ok: false, error: String(err) });
  }
}

function jsonResponse(payload) {
  return ContentService.createTextOutput(JSON.stringify(payload)).setMimeType(
    ContentService.MimeType.JSON
  );
}
