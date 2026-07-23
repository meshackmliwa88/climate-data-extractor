function updateUTCClock() {
    const timeBox = document.getElementById("utc-clock-time");
    const dateBox = document.getElementById("utc-clock-date");

    if (!timeBox || !dateBox) return;

    const now = new Date();

    const time = now.toLocaleTimeString("en-GB", {
        timeZone: "UTC",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false
    });

    const date = now.toLocaleDateString("en-GB", {
        timeZone: "UTC",
        weekday: "long",
        year: "numeric",
        month: "short",
        day: "2-digit"
    });

    timeBox.textContent = time + " UTC";
    dateBox.textContent = date.replace(",", "  •");
}

updateUTCClock();
setInterval(updateUTCClock, 1000);
