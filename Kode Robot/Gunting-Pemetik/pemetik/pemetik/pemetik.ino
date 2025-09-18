#include <WiFi.h>
#include <WebServer.h>


const char* ssid = "bismillah";
const char* password = "bunga123";
WebServer server(80);


const int IN3 = 14;
const int IN4 = 12;
const int ENA = 13;
const unsigned long T_45 = 120;  


enum MotorState { IDLE, STEP_FWD, STEP_BWD, CONT_FWD, CONT_BWD };
MotorState state = IDLE;
unsigned stepsQueued = 0;
bool currentDirection = true; 
unsigned long tStart = 0;

int loopCounter = 0;
bool loopRunning = false;
String currentStatus = "IDLE";


void handleRoot();
void handleCmd();
void handleStatus();
void startStep(bool forward);
void startContinuous(bool forward);
void stopMotor();
void processMotorQueue();


void setup() {
  Serial.begin(115200);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT);
  digitalWrite(ENA, HIGH);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected! IP: " + WiFi.localIP().toString());

  server.on("/", handleRoot);
  server.on("/cmd", handleCmd);
  server.on("/status", handleStatus);
  server.begin();
}


void loop() {
  server.handleClient();
  processMotorQueue();
}

void processMotorQueue() {
  unsigned long now = millis();

  switch (state) {
    case STEP_FWD:
    case STEP_BWD:
      if (now - tStart >= T_45) {
        stopMotor();
        if (stepsQueued > 0) {
          stepsQueued--;
          if (stepsQueued > 0) {
            // Lanjut ke step berikutnya
            startStep(currentDirection);
          } else if (loopRunning) {
            // Proses loop
            loopCounter++;
            if (loopCounter < 10) { 
              delay(100);
              currentDirection = !currentDirection;
              stepsQueued = 1;
              startStep(currentDirection);
            } else {
              loopRunning = false;
              loopCounter = 0;
              state = IDLE;
              currentStatus = "IDLE";
            }
          } else {
            state = IDLE;
            currentStatus = "IDLE";
          }
        } else {
          state = IDLE;
          currentStatus = "IDLE";
        }
      }
      break;

    case CONT_FWD:
    case CONT_BWD:
      
      break;

    case IDLE:
      // Tidak ada aksi yang sedang berlangsung
      break;
  }
}


void handleRoot() {
  String html = R"rawliteral(
<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>Motor 45° Controller</title>
<style>
  body{display:flex;flex-direction:column;align-items:center;gap:10px;
       font-family:Arial;background:#f7f9fc;margin:0;padding:40px}
  h2{margin-bottom:10px} .btn{width:160px;height:50px;font-size:16px;
       border:none;border-radius:10px;cursor:pointer;font-weight:bold}
  .green{background:#4caf50;color:#fff} .red{background:#f44336;color:#fff}
  .blue{background:#2196f3;color:#fff} .grey{background:#9e9e9e;color:#fff}
</style></head><body>
  <h2>Kontrol Motor DC</h2>
  <div id="status" style="margin-bottom:20px;font-size:18px;color:#333">Status: IDLE</div>

  <button class='btn green' onclick="send('f45')">↺ +45°</button>
  <button class='btn green' onclick="send('f90')">↺ +90°</button>
  <button class='btn blue'  onclick="send('cont_fwd')">⏩ Menutup Pemetik</button>
  <button class='btn grey'  onclick="send('stop')">⏹ STOP</button>
  <button class='btn blue'  onclick="send('cont_bwd')">⏪ Membuka Pemetik</button>
  <button class='btn red'   onclick="send('b45')">↻ −45°</button>
  <button class='btn red'   onclick="send('b90')">↻ −90°</button>
  <button class='btn grey'  onclick="send('loop5')">↔ Loop 5x</button>

<script>
function send(cmd){
  fetch('/cmd?motor=' + cmd)
    .then(() => console.log('Sent:', cmd));
}

setInterval(() => {
  fetch('/status')
    .then(res => res.text())
    .then(txt => {
      document.getElementById('status').innerText = 'Status: ' + txt;
    });
}, 500);
</script>
</body></html>
)rawliteral";
  server.send(200, "text/html", html);
}


void handleCmd() {
  String cmd = server.arg("motor");
  Serial.println("CMD: " + cmd);

  if (cmd == "f45") {
    if (state == IDLE) {
      stepsQueued = 1;
      currentDirection = true;
      startStep(true);
    } else if (state == STEP_FWD || state == STEP_BWD) {
      stepsQueued += 1;
    }
    currentStatus = "+45°";
  }
  else if (cmd == "b45") {
    if (state == IDLE) {
      stepsQueued = 1;
      currentDirection = false;
      startStep(false);
    } else if (state == STEP_FWD || state == STEP_BWD) {
      stepsQueued += 1;
    }
    currentStatus = "-45°";
  }
  else if (cmd == "f90") {
    if (state == IDLE) {
      stepsQueued = 2;
      currentDirection = true;
      startStep(true);
    } else if (state == STEP_FWD || state == STEP_BWD) {
      stepsQueued += 2;
    }
    currentStatus = "+90°";
  }
  else if (cmd == "b90") {
    if (state == IDLE) {
      stepsQueued = 2;
      currentDirection = false;
      startStep(false);
    } else if (state == STEP_FWD || state == STEP_BWD) {
      stepsQueued += 2;
    }
    currentStatus = "-90°";
  }
  else if (cmd == "cont_fwd") {
    startContinuous(true);
    currentStatus = "Menutup Pemetik";
  }
  else if (cmd == "cont_bwd") {
    startContinuous(false);
    currentStatus = "Membuka Pemetik";
  }
  else if (cmd == "stop") {
    stepsQueued = 0;
    stopMotor();
    loopRunning = false;
    loopCounter = 0;
    state = IDLE;
    currentStatus = "IDLE";
  }
  else if (cmd == "loop5") {
    if (!loopRunning && state == IDLE) {
      loopRunning = true;
      loopCounter = 0;
      stepsQueued = 1;
      currentDirection = true;
      startStep(true);
      currentStatus = "Loop 5x";
    }
  }

  server.send(200, "text/plain", "OK");
}


void handleStatus() {
  server.send(200, "text/plain", currentStatus);
}


void startStep(bool forward) {
  if (forward) {
    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
    state = STEP_FWD;
  } else {
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, HIGH);
    state = STEP_BWD;
  }
  tStart = millis();
}

void startContinuous(bool forward) {
  stepsQueued = 0;
  loopRunning = false;
  if (forward) {
    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
    state = CONT_FWD;
  } else {
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, HIGH);
    state = CONT_BWD;
  }
}

void stopMotor() {
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}
