"use strict";
const $ = id => document.getElementById(id);
let snapshot = null, replays = {}, refreshing = false, running = false;
let token = new URLSearchParams(location.search).get("token") || localStorage.getItem("live_feed_phone_token") || "";
if (location.search) history.replaceState(null, "", location.pathname + location.hash);
const money = value => Number(value).toLocaleString(undefined, {style:"currency", currency:"USD", maximumFractionDigits:2});
const quantity = value => Number(value).toLocaleString(undefined, {maximumFractionDigits:10});
const percent = value => value == null ? "Unavailable" : `${Number(value).toFixed(2)}%`;
const stamp = value => value ? new Date(value).toLocaleString() : "Unavailable";
const words = code => String(code).replaceAll("_", " ");
const age = seconds => seconds == null ? "Age unknown" : seconds < 60 ? `${Math.round(seconds)}s old` : seconds < 3600 ? `${Math.floor(seconds / 60)}m old` : `${(seconds / 3600).toFixed(1)}h old`;
function node(tag, text, className) { const n = document.createElement(tag); if (text != null) n.textContent = text; if(className)n.className=className; return n; }
function cell(row, text, className) { row.append(node("td", text, className)); }
function showError(text) { $("error").textContent = text; $("error").hidden = !text; }
async function api(path, body) {
  const response = await fetch(path, {method:body ? "POST" : "GET", headers:{...(token ? {"X-Live-Feed-Token":token} : {}), ...(body ? {"Content-Type":"application/json"} : {})}, body:body ? JSON.stringify(body) : undefined, cache:"no-store", signal:AbortSignal.timeout(30000)});
  const data = await response.json();
  if(response.status===401) { $("auth").hidden=false; throw new Error("Dashboard token required. Your Coinbase credentials are not needed here."); }
  if(!response.ok) throw new Error(words(data.error || "request_failed"));
  return data;
}
function selectView(view) {
  if (!["portfolio","practice","performance","readiness"].includes(view)) view="portfolio";
  document.querySelectorAll("[data-view]").forEach(b=> { const active=b.dataset.view===view; b.setAttribute("aria-selected",String(active)); b.tabIndex=active?0:-1; $(b.dataset.view).hidden=!active; });
  history.replaceState(null,"",`${location.pathname}#${view}`);
  if(view==="performance") requestAnimationFrame(renderResult);
}
document.querySelectorAll("[data-view]").forEach(b=> {
  b.addEventListener("click",()=>selectView(b.dataset.view));
  b.addEventListener("keydown",event=>{ const tabs=Array.from(document.querySelectorAll("[data-view]")); let i=tabs.indexOf(b); if(event.key==="ArrowRight")i=(i+1)%tabs.length; else if(event.key==="ArrowLeft")i=(i+tabs.length-1)%tabs.length; else if(event.key==="Home")i=0; else if(event.key==="End")i=tabs.length-1; else return; event.preventDefault();selectView(tabs[i].dataset.view);tabs[i].focus(); });
});
document.querySelectorAll("[data-go]").forEach(b=>b.addEventListener("click",()=>selectView(b.dataset.go)));
$("auth").addEventListener("submit",event=>{event.preventDefault();token=$("token").value.trim();localStorage.setItem("live_feed_phone_token",token);refresh();});
function renderPortfolio() {
  const p=snapshot.portfolio;
  const btc=p.balances.find(row=>row.currency==="BTC");
  $("btc-balance").textContent=btc ? `${quantity(btc.total)} BTC` : "Unavailable";
  $("balance-time").textContent=`Snapshot: ${stamp(p.verified_at)}`;
  $("portfolio-notice").textContent=p.fresh ? "Verified account snapshot. Available and held amounts are cached Coinbase values." : p.configured ? "Cached holdings, not a current balance. Account verification is refreshed through the existing Coinbase account-link command." : "No verified account snapshot is available to this dashboard. No credentials are requested here.";
  $("balances").replaceChildren();
  const rows=p.balances.filter(row=>!$("btc-only").checked || row.currency==="BTC");
  rows.forEach(row=>{const tr=node("tr");cell(tr,row.currency);cell(tr,quantity(row.available),"number");cell(tr,quantity(row.hold),"number");cell(tr,quantity(row.total),"number");$("balances").append(tr);});
  if(!rows.length){const tr=node("tr"),td=node("td","No matching verified balance rows.");td.colSpan=4;tr.append(td);$("balances").append(tr);}
}
$("btc-only").addEventListener("change",()=>snapshot && renderPortfolio());
function gate(title, state, detail, style="amber-bg") { const div=node("div",null,"gate");div.append(node("strong",title),node("span",state,`badge ${style}`),node("p",detail));return div; }
function renderSchwab() {
  const s=snapshot.schwab_context;
  $("schwab-quotes").replaceChildren();
  $("schwab-state").textContent=s?words(s.state):"Not collected";
  $("schwab-time").textContent=s?`${s.usable_instruments} usable instruments / producer ${stamp(s.evidence.timestamp_utc)}${s.warnings.length?" / "+s.warnings.map(words).join("; "):""}`:"Collector evidence unavailable";
  (s?.instruments || []).forEach(row=>{const tr=node("tr");cell(tr,row.asset);cell(tr,row.source_symbol || row.requested_symbol);cell(tr,row.instrument_type==="future"?"Futures":"Fund / ETP");cell(tr,row.mark==null?"Unavailable":money(row.mark),"number");cell(tr,row.spread_bps==null?"Unavailable":Number(row.spread_bps).toFixed(2),"number");cell(tr,age(row.age_seconds));cell(tr,words(row.quality),row.usable?"":"amber");$("schwab-quotes").append(tr);});
  if(!s?.instruments.length){const tr=node("tr"),td=node("td","No verified Schwab quote rows in the latest collector evidence.");td.colSpan=7;tr.append(td);$("schwab-quotes").append(tr);}
}
function renderSnapshot() {
  const {portfolio:p,collection:c,training:t,paper}=snapshot;
  $("account-state").textContent=p.fresh?"Verified":p.configured?"Cached / stale":"Unavailable";
  $("account-time").textContent=p.configured?age(p.age_seconds):"Read-only snapshot missing";
  $("collection-state").textContent=c.running?"Collecting":"Stale / unknown";
  $("collection-time").textContent=`${c.symbols ?? "Unknown"} crypto symbols / ${age(c.evidence.age_seconds)}`;
  $("training-state").textContent=t.passed!=null && t.assessed!=null?`${t.passed} / ${t.assessed} passed`:"Unassessed";
  $("training-time").textContent=`${t.samples ?? "Unknown"} materialized samples / ${age(t.label_evidence.age_seconds)}`;
  renderPortfolio();
  renderSchwab();
  $("practice-blockers").replaceChildren(...paper.blockers.map(x=>node("li",words(x))));
  const trainingDetail=[!t.evidence.fresh?"Training control evidence is stale or missing.":"",...t.blockers.map(words)].filter(Boolean).join("; ");
  $("gates").replaceChildren(
    gate("Live money & transfers","Locked","This section has no broker submission, transfer, or unlock endpoint.","locked"),
    gate("Historical research","Available","User-triggered BTC candle replay. No forward orders or acceptance credit.","green"),
    gate("Forward paper trading","Held",paper.blockers.map(words).join("; ")),
    gate("Training launch",t.launch_allowed?"Owner gate open":"Not cleared",trainingDetail || "No training is launched by this section."),
    gate("Account evidence",p.fresh?"Fresh":"Not current",`Last account verification: ${stamp(p.verified_at)}.`,p.fresh?"green":"amber-bg"),
    gate("Collection evidence",c.running?"Fresh":"Not current",`Producer timestamp: ${stamp(c.evidence.timestamp_utc)}.`,c.running?"green":"amber-bg")
  );
  $("label-time").textContent=`Assessed ${stamp(t.label_evidence.timestamp_utc)}${t.label_evidence.fresh?"":" / stale"}`;
  $("training-bots").replaceChildren();
  t.bots.forEach(bot=>{const tr=node("tr");cell(tr,words(bot.bot_id.replace("brain_refinery_","")));cell(tr,bot.observations ?? "Unavailable","number");cell(tr,bot.data_checks_passed?"Passed":"Blocked");cell(tr,bot.blockers.map(words).join("; ") || "None reported");$("training-bots").append(tr);});
  $("updated").textContent=`Evidence read ${stamp(snapshot.timestamp_utc)}`;
}
async function refresh() {
  if(refreshing)return;refreshing=true;$("refresh").disabled=true;
  try { snapshot=await api("/api/crypto/status");renderSnapshot();$("auth").hidden=true;showError(""); }
  catch(error) { showError(error.message);$("updated").textContent="Evidence reload failed";$("account-state").textContent="Refresh failed";$("collection-state").textContent="Unverified";if(snapshot){snapshot.portfolio.fresh=false;renderPortfolio();if(snapshot.schwab_context){snapshot.schwab_context.state="refresh_failed";snapshot.schwab_context.usable_instruments=0;snapshot.schwab_context.instruments.forEach(row=>{row.usable=false;row.quality="unverified";});renderSchwab();}} }
  finally {refreshing=false;$("refresh").disabled=false;}
}
$("refresh").addEventListener("click",refresh);
document.querySelectorAll(".replay-form").forEach(form=>form.addEventListener("submit",async event=>{
  event.preventDefault();if(running)return;
  const body={profile:form.dataset.profile}; for(const [key,value] of new FormData(form))body[key]=Number(value);
  const status=form.querySelector(".run-status");running=true;document.querySelectorAll(".replay-form button").forEach(b=>b.disabled=true);
  status.textContent="Loading closed BTC candles and running historical research...";
  try { const result=await api("/api/crypto/replay",body);replays[body.profile]=result;status.textContent=`Complete: ${result.round_trips} round trips. Research only.`;$("result-profile").value=body.profile;selectView("performance"); }
  catch(error){status.textContent=`Replay not completed: ${error.message}. No results were generated.`;}
  finally {running=false;document.querySelectorAll(".replay-form button").forEach(b=>b.disabled=false);}
}));
function metric(label,value,detail) {const d=node("div");d.append(node("span",label,"metric-label"),node("strong",value),node("small",detail));return d;}
function renderResult() {
  const r=replays[$("result-profile").value];$("no-results").hidden=!!r;$("results").hidden=!r;$("download").disabled=!r;if(!r)return;
  $("result-period").textContent=`Historical research / ${stamp(r.start*1000)} to ${stamp(r.end*1000)} / ${r.candles} closed bars, ${r.warmup_bars} warmup. Not forward-paper performance.`;
  $("result-metrics").replaceChildren(metric("NET P&L",money(r.net_pnl),`${percent(r.return_pct)} after costs`),metric("MAX DRAWDOWN",percent(r.max_drawdown_pct),`${r.round_trips} closed round trips`),metric("MODELED COSTS",money(r.fees+r.friction),`${money(r.fees)} fees + ${money(r.friction)} friction`),metric("BUY & HOLD",percent(r.benchmark_return_pct),"Same evaluation window, after costs"));
  $("result-assumptions").textContent=`Virtual start ${money(r.initial)} / end ${money(r.ending)}. Win rate ${percent(r.win_rate_pct)}. ${r.assumptions.join(". ")}. Source SHA-256 ${r.data_sha256.slice(0,16)}. Results remain in server memory until restart; export preserves this research record.`;
  $("fills").replaceChildren();r.fills.forEach(f=>{const tr=node("tr");cell(tr,new Date(f.time*1000).toISOString().slice(0,16).replace("T"," "));cell(tr,f.action);cell(tr,quantity(f.quantity),"number");cell(tr,money(f.price),"number");cell(tr,money(f.fee),"number");cell(tr,f.reason);$("fills").append(tr);});
  $("fill-count").textContent=`${r.fills.length} modeled fills`;drawChart(r);
}
function drawChart(r) {
  const canvas=$("equity-chart"),width=canvas.clientWidth,height=canvas.clientHeight;if(!width)return;
  const scale=window.devicePixelRatio||1;canvas.width=Math.round(width*scale);canvas.height=Math.round(height*scale);const ctx=canvas.getContext("2d");ctx.scale(scale,scale);ctx.clearRect(0,0,width,height);
  const left=65,right=16,top=25,bottom=32,values=[r.initial,...r.curve.map(p=>p.equity),...r.benchmark_curve.map(p=>p.equity)];
  let min=Math.min(...values),max=Math.max(...values),padding=Math.max((max-min)*.15,r.initial*.002);min-=padding;max+=padding;
  ctx.font="11px -apple-system, sans-serif";ctx.textAlign="right";ctx.fillStyle="#65716c";ctx.strokeStyle="#e5eae7";ctx.lineWidth=1;
  for(let i=0;i<5;i++){const y=top+(height-top-bottom)*i/4;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(width-right,y);ctx.stroke();ctx.fillText(money(max-(max-min)*i/4),left-10,y+4);}
  for(const [points,color,dash] of [[r.benchmark_curve,"#796dab",[5,4]],[r.curve,"#14765c",[]]]){ctx.strokeStyle=color;ctx.lineWidth=2;ctx.setLineDash(dash);ctx.beginPath();points.forEach((p,i)=>{const x=left+(width-left-right)*i/Math.max(1,points.length-1),y=top+(height-top-bottom)*(max-p.equity)/(max-min);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();}
  ctx.setLineDash([]);ctx.fillStyle="#65716c";ctx.textAlign="left";ctx.fillText(new Date(r.start*1000).toLocaleDateString(),left,height-10);ctx.textAlign="right";ctx.fillText(new Date(r.end*1000).toLocaleDateString(),width-right,height-10);
}
$("result-profile").addEventListener("change",renderResult);
new ResizeObserver(()=>{if(!$("performance").hidden)renderResult();}).observe($("equity-chart"));
$("download").addEventListener("click",()=>{const r=replays[$("result-profile").value];if(!r)return;const a=node("a");const params=new URLSearchParams();if(token)params.set("token",token);a.href=`/api/crypto/export/${r.profile}${token?"?"+params.toString():""}`;a.download=`btc-${r.profile}-historical-research.json`;a.rel="noreferrer";document.body.append(a);a.click();a.remove();});
selectView(location.hash.slice(1));refresh();
api("/api/crypto/replays").then(data=>{replays=data;renderResult();}).catch(()=>{});
setInterval(()=>{if(!document.hidden)refresh();},30000);
