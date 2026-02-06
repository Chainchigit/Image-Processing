/*********************************
 * CONFIGURATION
 *********************************/
const CONFIG = {
  TZ: 'Asia/Bangkok',
  LINE_TOKEN: 'xxx',
  TARGET_ID: 'xxx', // แก้เป็น ID ที่ต้องการส่ง
  ROOT_ID: 'xxx',
  RELAY_SECRET: 'xxx', // รหัสสำหรับ Webhook URL

  CAMS: ['cam1', 'cam2'],
  CLASSES: ['good', 'hole', 'low', 'oxy', 'reject']
};

/*********************************
 * 1. REALTIME MONITORING (เวอร์ชันอัปเดต Logic)
 *********************************/
function tick_sendLeafSummaryRealtime() {
  const today = getToday_();
  const root = DriveApp.getFolderById(CONFIG.ROOT_ID);
  const dayFolder = findFolder_(root, today);
  
  // ตรวจสอบว่ามีโฟลเดอร์ของวันนี้ไหม
  if (!dayFolder) {
    sendLine_(`⚠️ ไม่พบโฟลเดอร์สำหรับวันที่ ${today}\n(ระบบหยุดการนับไฟล์ชั่วคราว)`);
    return;
  }

  const props = PropertiesService.getScriptProperties();
  const stateKey = `STATE_${today}`;

  const prevState = loadState_(props, stateKey);
  const nowState  = buildState_(dayFolder);

  // ตรวจสอบว่าใน Drive มีรูปภาพรวมกันบ้างไหม
  if (nowState.total === 0) {
    sendLine_(`⚠️ ไม่มีไฟล์ image ใน drive สำหรับวันนี้ (${today})`);
    saveState_(props, stateKey, nowState);
    return;
  }

  const diff = calcDiff_(prevState, nowState);

  // ถ้าไม่มีไฟล์ใหม่เพิ่มขึ้น ไม่ต้องส่งข้อความซ้ำ
  if (diff.totalNew === 0) {
    saveState_(props, stateKey, nowState);
    return;
  }

  const message = buildLineMessage_(today, diff, nowState, dayFolder.getUrl());
  sendLine_(message);
  saveState_(props, stateKey, nowState);
}

/*********************************
 * LINE MESSAGE (เวอร์ชันคำนวณ % Defect)
 *********************************/
function buildLineMessage_(today, diff, nowState, folderUrl) {
  const lines = [
    `🌿 *Leaf Realtime Update* (${today})`,
    `➕ New files: ${diff.totalNew}`,
    ''
  ];

  CONFIG.CAMS.forEach(cam => {
    const camData = diff.camDiff[cam];
    if (!camData || camData.totalNew === 0) return;

    lines.push(`📷 ${cam.toUpperCase()} (+${camData.totalNew})`);
    CONFIG.CLASSES.forEach(cls => {
      if (camData.classes[cls] > 0) {
        lines.push(`  - ${cls}: +${camData.classes[cls]}`);
      }
    });
    lines.push('');
  });

  lines.push(`📊 *Defect Rate Summary*`);
  
  CONFIG.CAMS.forEach(cam => {
    const c = nowState.cams[cam];
    if (!c || c.total === 0) {
      lines.push(`  - ${cam}: 0 files`);
      return;
    }

    // คำนวณ % Defect: (hole + oxy + low) / total_cam
    const defectCount = (c.classes['hole'] || 0) + (c.classes['oxy'] || 0) + (c.classes['low'] || 0);
    const defectRate = ((defectCount / c.total) * 100).toFixed(2);
    
    lines.push(`  - ${cam}: ${c.total} files (Defect: ${defectRate}%)`);
  });

  lines.push(`\n📈 Total overall: ${nowState.total}`);
  lines.push('', `🔗 Folder: ${folderUrl}`);
  return lines.join('\n');
}

/*********************************
 * 2. WEBHOOK HANDLING (รับค่าจาก LINE)
 *********************************/
function doPost(e) {
  try {
    const secret = e?.parameter?.secret || "";
    if (secret !== CONFIG.RELAY_SECRET) {
      return ContentService.createTextOutput("forbidden");
    }

    const raw = e?.postData?.contents || "{}";
    const props = PropertiesService.getScriptProperties();
    
    // เก็บข้อมูลดิบลง Property
    props.setProperty("PENDING_WEBHOOK", raw);

    // สร้าง Time-based trigger ทำงานทันทีหลังจากนี้ 1 วินาที (เพื่อตอบ LINE ให้ไวที่สุด)
    deleteOldTriggers_("processWebhook_"); // เคลียร์ของเก่าถ้ามีค้าง
    ScriptApp.newTrigger("processWebhook_")
      .timeBased()
      .after(1000)
      .create();

    return ContentService.createTextOutput("ok").setMimeType(ContentService.MimeType.TEXT);
  } catch (err) {
    return ContentService.createTextOutput("error").setMimeType(ContentService.MimeType.TEXT);
  }
}

// ฟังก์ชันประมวลผล Webhook เบื้องหลัง
function processWebhook_() {
  const props = PropertiesService.getScriptProperties();
  const raw = props.getProperty("PENDING_WEBHOOK");
  if (!raw) return;

  props.deleteProperty("PENDING_WEBHOOK");
  deleteOldTriggers_("processWebhook_");

  let data;
  try { data = JSON.parse(raw); } catch (e) { return; }

  const ev = (data.events && data.events[0]) ? data.events[0] : null;
  if (!ev) return;

  const src = ev.source || {};

  // เก็บ ID อัตโนมัติเพื่อให้ Admin มาเปิดดูใน Script Properties
  if (src.userId)  props.setProperty("LAST_USER_ID", src.userId);
  if (src.groupId) props.setProperty("LAST_GROUP_ID", src.groupId);
  if (src.roomId)  props.setProperty("LAST_ROOM_ID", src.roomId);
  
  props.setProperty("LAST_EVENT_JSON", JSON.stringify(ev, null, 2));
}

/*********************************
 * CORE LOGIC & HELPERS
 *********************************/

function buildState_(dayFolder) {
  const state = { cams: {}, total: 0, updatedAt: new Date().toISOString() };
  CONFIG.CAMS.forEach(cam => {
    const camFolder = findFolder_(dayFolder, cam);
    if (!camFolder) return;

    const camState = { classes: {}, total: 0 };
    CONFIG.CLASSES.forEach(cls => {
      const clsFolder = findFolder_(camFolder, cls);
      const count = clsFolder ? countFiles_(clsFolder) : 0;
      camState.classes[cls] = count;
      camState.total += count;
      state.total += count;
    });
    state.cams[cam] = camState;
  });
  return state;
}

function calcDiff_(prev = {}, now) {
  let totalNew = 0;
  const camDiff = {};

  CONFIG.CAMS.forEach(cam => {
    const prevClasses = prev.cams?.[cam]?.classes || {};
    const nowClasses  = now.cams?.[cam]?.classes  || {};
    const clsDiff = {};
    let camNew = 0;

    CONFIG.CLASSES.forEach(cls => {
      const diff = Math.max(0, (nowClasses[cls] || 0) - (prevClasses[cls] || 0));
      clsDiff[cls] = diff;
      camNew += diff;
      totalNew += diff;
    });
    camDiff[cam] = { classes: clsDiff, totalNew: camNew };
  });
  return { totalNew, camDiff };
}

/*********************************
 * LINE MESSAGE (เพิ่มสรุปภาพรวม Defect รวม)
 *********************************/
function buildLineMessage_(today, diff, nowState, folderUrl) {
  const lines = [
    `🌿 *Leaf Realtime Update* (${today})`,
    `➕ New files: ${diff.totalNew}`,
    ''
  ];

  // 1. รายงานไฟล์ใหม่ที่เพิ่มเข้ามา
  CONFIG.CAMS.forEach(cam => {
    const camData = diff.camDiff[cam];
    if (!camData || camData.totalNew === 0) return;

    lines.push(`📷 ${cam.toUpperCase()} (+${camData.totalNew})`);
    CONFIG.CLASSES.forEach(cls => {
      if (camData.classes[cls] > 0) {
        lines.push(`  - ${cls}: +${camData.classes[cls]}`);
      }
    });
    lines.push('');
  });

  lines.push(`📊 *Defect Rate Summary*`);
  
  let grandTotalFiles = 0;
  let grandTotalDefects = 0;

  // 2. รายงานสรุปแยกรายกล้อง
  CONFIG.CAMS.forEach(cam => {
    const c = nowState.cams[cam];
    if (!c || c.total === 0) {
      lines.push(`  - ${cam}: 0 files`);
      return;
    }

    const hole = c.classes['hole'] || 0;
    const oxy  = c.classes['oxy'] || 0;
    const low  = c.classes['low'] || 0;
    
    const camDefectCount = hole + oxy + low;
    const camDefectRate = ((camDefectCount / c.total) * 100).toFixed(2);
    
    // สะสมยอดรวมสำหรับ Grand Total
    grandTotalFiles += c.total;
    grandTotalDefects += camDefectCount;
    
    lines.push(`  - ${cam}: ${c.total} files (Defect: ${camDefectRate}%)`);
  });

  // 3. คำนวณภาพรวมทั้งหมด (Grand Total)
  const grandDefectRate = grandTotalFiles > 0 
    ? ((grandTotalDefects / grandTotalFiles) * 100).toFixed(2) 
    : "0.00";

  lines.push(`\n📈 *Total overall*`);
  lines.push(`  - Total files: ${grandTotalFiles}`);
  lines.push(`  - Total Defect: ${grandDefectRate}%`); // <--- เพิ่มตรงนี้ตามที่ต้องการ
  
  lines.push('', `🔗 Folder: ${folderUrl}`);
  
  return lines.join('\n');
}

function sendLine_(text) {
  if (CONFIG.TARGET_ID === 'PUT_USER_OR_GROUP_ID') return;
  UrlFetchApp.fetch('https://api.line.me/v2/bot/message/push', {
    method: 'post',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${CONFIG.LINE_TOKEN}`
    },
    payload: JSON.stringify({
      to: CONFIG.TARGET_ID,
      messages: [{ type: 'text', text }]
    }),
    muteHttpExceptions: true
  });
}

function getToday_() {
  return Utilities.formatDate(new Date(), CONFIG.TZ, 'yyyy-MM-dd');
}

function findFolder_(parent, name) {
  const it = parent.getFoldersByName(name);
  return it.hasNext() ? it.next() : null;
}

function countFiles_(folder) {
  let count = 0;
  const it = folder.getFiles();
  while (it.hasNext()) { it.next(); count++; }
  return count;
}

function loadState_(props, key) {
  try {
    const val = props.getProperty(key);
    return val ? JSON.parse(val) : {};
  } catch (e) { return {}; }
}

function saveState_(props, key, state) {
  props.setProperty(key, JSON.stringify(state));
}

function deleteOldTriggers_(functionName) {
  const triggers = ScriptApp.getProjectTriggers();
  triggers.forEach(t => {
    if (t.getHandlerFunction() === functionName) ScriptApp.deleteTrigger(t);
  });
}



/*********************************
 * 3. DAILY SUMMARY REPORT (ส่งตอน 17:00)
 *********************************/
function sendDailyReport() {
  const today = getToday_();
  const root = DriveApp.getFolderById(CONFIG.ROOT_ID);
  const dayFolder = findFolder_(root, today);
  
  if (!dayFolder) return;

  const nowState = buildState_(dayFolder);
  if (nowState.total === 0) return;

  let grandTotalFiles = 0;
  let grandTotalBad = 0;
  
  let totalGood = 0;
  let totalHole = 0;
  let totalLow = 0;
  let totalOxy = 0;
  let totalReject = 0;
  
  let rows = [];

  CONFIG.CAMS.forEach(cam => {
    const c = nowState.cams[cam];
    if (!c) return;

    const total = c.total || 0;
    const good = c.classes['good'] || 0;
    const hole = c.classes['hole'] || 0;
    const low = c.classes['low'] || 0;
    const oxy = c.classes['oxy'] || 0;
    const reject = c.classes['reject'] || 0;
    
    const camBad = hole + low + oxy;
    
    grandTotalFiles += total;
    grandTotalBad += camBad;
    totalGood += good;
    totalHole += hole;
    totalLow += low;
    totalOxy += oxy;
    totalReject += reject;

    rows.push({
      "type": "box", "layout": "horizontal", "margin": "sm", "contents": [
        { "type": "text", "text": cam.toUpperCase(), "size": "xxs", "weight": "bold", "flex": 2 },
        { "type": "text", "text": total.toString(), "size": "xxs", "align": "center", "flex": 1 },
        { "type": "text", "text": good.toString(), "size": "xxs", "align": "center", "color": "#06d6a0", "flex": 1 },
        { "type": "text", "text": hole.toString(), "size": "xxs", "align": "center", "color": "#ef476f", "flex": 1 },
        { "type": "text", "text": low.toString(), "size": "xxs", "align": "center", "color": "#ef476f", "flex": 1 },
        { "type": "text", "text": oxy.toString(), "size": "xxs", "align": "center", "color": "#ef476f", "flex": 1 },
        { "type": "text", "text": reject.toString(), "size": "xxs", "align": "center", "color": "#118ab2", "flex": 1 }
      ]
    });
  });

  const grandDefectRate = grandTotalFiles > 0 
    ? ((grandTotalBad / grandTotalFiles) * 100).toFixed(2) 
    : "0.00";

  // แถว TOTAL สรุปรายคอลัมน์ (ใช้ขนาด xxs เพื่อประหยัดพื้นที่)
  rows.push({ "type": "separator", "margin": "md" });
  rows.push({
    "type": "box", "layout": "horizontal", "margin": "sm", "contents": [
      { "type": "text", "text": "TOTAL", "size": "xxs", "weight": "bold", "flex": 2, "color": "#aaaaaa" },
      { "type": "text", "text": grandTotalFiles.toString(), "size": "xxs", "align": "center", "flex": 1, "weight": "bold" },
      { "type": "text", "text": totalGood.toString(), "size": "xxs", "align": "center", "flex": 1, "weight": "bold", "color": "#06d6a0" },
      { "type": "text", "text": totalHole.toString(), "size": "xxs", "align": "center", "flex": 1, "weight": "bold", "color": "#ef476f" },
      { "type": "text", "text": totalLow.toString(), "size": "xxs", "align": "center", "flex": 1, "weight": "bold", "color": "#ef476f" },
      { "type": "text", "text": totalOxy.toString(), "size": "xxs", "align": "center", "flex": 1, "weight": "bold", "color": "#ef476f" },
      { "type": "text", "text": totalReject.toString(), "size": "xxs", "align": "center", "flex": 1, "weight": "bold", "color": "#118ab2" }
    ]
  });

  const flexPayload = {
    "type": "bubble",
    "size": "giga",
    "header": {
      "type": "box", "layout": "vertical", "backgroundColor": "#073b4c", "paddingAll": "md", "contents": [
        { "type": "text", "text": "DAILY QUALITY REPORT", "weight": "bold", "color": "#ffffff", "size": "sm" },
        { "type": "text", "text": today, "color": "#ffffff", "size": "xxs" }
      ]
    },
    "body": {
      "type": "box", "layout": "vertical", "paddingAll": "sm", "contents": [
        { "type": "box", "layout": "horizontal", "contents": [
            { "type": "text", "text": "CAM", "size": "xxs", "color": "#aaaaaa", "weight": "bold", "flex": 2 },
            { "type": "text", "text": "TOT", "size": "xxs", "color": "#aaaaaa", "weight": "bold", "flex": 1, "align": "center" },
            { "type": "text", "text": "GD", "size": "xxs", "color": "#aaaaaa", "weight": "bold", "flex": 1, "align": "center" },
            { "type": "text", "text": "HL", "size": "xxs", "color": "#aaaaaa", "weight": "bold", "flex": 1, "align": "center" },
            { "type": "text", "text": "LO", "size": "xxs", "color": "#aaaaaa", "weight": "bold", "flex": 1, "align": "center" },
            { "type": "text", "text": "OX", "size": "xxs", "color": "#aaaaaa", "weight": "bold", "flex": 1, "align": "center" },
            { "type": "text", "text": "RJ", "size": "xxs", "color": "#aaaaaa", "weight": "bold", "flex": 1, "align": "center" }
          ]
        },
        { "type": "separator", "margin": "xs" },
        { "type": "box", "layout": "vertical", "contents": rows },
        { "type": "separator", "margin": "xl" },
        { "type": "box", "layout": "vertical", "spacing": "xs", "paddingAll": "sm", "contents": [
            { "type": "box", "layout": "horizontal", "contents": [
                { "type": "text", "text": "TOTAL FILES", "size": "xs", "color": "#555555", "flex": 4 },
                { "type": "text", "text": grandTotalFiles.toString(), "size": "xs", "align": "end", "weight": "bold", "flex": 2 }
              ]
            },
            { "type": "box", "layout": "horizontal", "contents": [
                { "type": "text", "text": "TOTAL BAD", "size": "xs", "color": "#555555", "flex": 4 },
                { "type": "text", "text": grandTotalBad.toString(), "size": "xs", "align": "end", "weight": "bold", "color": "#ef476f", "flex": 2 }
              ]
            },
            { "type": "box", "layout": "horizontal", "contents": [
                { "type": "text", "text": "% TOTAL DEFECT", "size": "xs", "color": "#555555", "flex": 4 },
                { "type": "text", "text": grandDefectRate + "%", "size": "xs", "align": "end", "weight": "bold", "color": "#ef476f", "flex": 2 }
              ]
            }
          ]
        }
      ]
    },
    "footer": {
      "type": "box", "layout": "vertical", "contents": [
        { "type": "button", "action": { "type": "uri", "label": "VIEW GOOGLE DRIVE", "uri": dayFolder.getUrl() }, "style": "primary", "color": "#118ab2", "height": "sm" }
      ]
    }
  };

  sendFlex_(flexPayload);
}

function sendFlex_(payload) {
  UrlFetchApp.fetch('https://api.line.me/v2/bot/message/push', {
    method: 'post',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${CONFIG.LINE_TOKEN}`
    },
    payload: JSON.stringify({
      to: CONFIG.TARGET_ID,
      messages: [{
        "type": "flex",
        "altText": "Daily Report Summary",
        "contents": payload
      }]
    }),
    muteHttpExceptions: true
  });
}
