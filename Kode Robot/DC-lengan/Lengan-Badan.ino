#include <WiFi.h>
#include <WebServer.h>


const char* ssid     = "bismillah";
const char* password = "bunga123";
WebServer server(80);

/* atas-bawah lengan */
const int IN3 = 14;
const int IN4 = 12;
const int ENA = 13;

/* kanan-kiri, badan */
const int IN1 = 27;
const int IN2 = 26;
const int ENB = 25;

const unsigned long T_45 = 120;


enum MotorState { 
  IDLE, 
  STEP_UP, STEP_DOWN, CONT_UP, CONT_DOWN,  // Motor 1 - Atas/Bawah
  STEP_LEFT, STEP_RIGHT, CONT_LEFT, CONT_RIGHT  // Motor 2 - Kiri/Kanan
};
MotorState state = IDLE;
unsigned queueUp = 0;
unsigned queueDown = 0;
unsigned queueLeft = 0;
unsigned queueRight = 0;
unsigned long tStart = 0;

int loopCounter = 0;
bool loopRunning = false;
String currentStatus = "IDLE";

// Function declarations
void handleRoot();
void handleCmd();
void handleStatus();
void startStep(bool up, bool horizontal);
void startContinuous(bool up, bool horizontal);
void stopAllMotors();
void stopMotor1();
void stopMotor2();


void setup() {
  Serial.begin(115200);
  
  // Setup Motor 1 (Atas-Bawah)
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT);
  digitalWrite(ENA, HIGH);
  
  // Setup Motor 2 (Kanan-Kiri)
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT);
  digitalWrite(ENB, HIGH);

  // Menghapus WiFi.config() untuk menggunakan DHCP
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

/*  LOOP */
void loop() {
  server.handleClient();
  unsigned long now = millis();

  switch (state) {
    case STEP_UP:
    case STEP_DOWN:
    case STEP_LEFT:
    case STEP_RIGHT:
      if (now - tStart >= T_45) {
        if (state == STEP_UP || state == STEP_DOWN) {
          stopMotor1();
          if (state == STEP_UP && queueUp > 0) queueUp--;
          else if (state == STEP_DOWN && queueDown > 0) queueDown--;
        } else {
          stopMotor2();
          if (state == STEP_LEFT && queueLeft > 0) queueLeft--;
          else if (state == STEP_RIGHT && queueRight > 0) queueRight--;
        }

        // Jika sedang loop 5x
        if (loopRunning) {
          loopCounter++;
          if (loopCounter < 10) {  
            delay(100);  // jeda kecil antara langkah
            startStep(loopCounter % 2 == 0, false); // ganjil = turun, genap = naik ,hanya vertikal saja
          } else {
            loopRunning = false;
            loopCounter = 0;
          }
        }

        state = IDLE;
      }
      break;

    case CONT_UP:
    case CONT_DOWN:
    case CONT_LEFT:
    case CONT_RIGHT:
      // Continuous movement - no automatic stop
      break;

    case IDLE:
    default:
      if (queueUp) startStep(true, false);
      else if (queueDown) startStep(false, false);
      else if (queueLeft) startStep(true, true);
      else if (queueRight) startStep(false, true);
      break;
  }
}

/*  HTML  */
void handleRoot() {
  String html = R"rawliteral(
<!DOCTYPE html><html><head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Kontrol Lengan Robot</title>
<style>
  body {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 15px;
    font-family: Arial, sans-serif;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    margin: 0;
    padding: 20px;
    min-height: 100vh;
  }
  
  .container {
    background: white;
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    max-width: 600px;
    width: 100%;
    text-align: center;
  }
  
  h2 {
    color: #2c3e50;
    margin-bottom: 15px;
  }
  
  .status-box {
    background: #ecf0f1;
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 20px;
    font-size: 18px;
    font-weight: bold;
    color: #2c3e50;
  }
  
  .control-section {
    margin-bottom: 25px;
  }
  
  .section-title {
    font-size: 18px;
    color: #2c3e50;
    margin-bottom: 12px;
    font-weight: bold;
    border-bottom: 2px solid #3498db;
    padding-bottom: 5px;
  }
  
  .btn-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    width: 100%;
    margin-bottom: 15px;
  }
  
  .btn {
    padding: 15px 10px;
    font-size: 16px;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-weight: bold;
    transition: all 0.3s ease;
  }
  
  .btn:active {
    transform: scale(0.98);
  }
  
  .btn-up {
    background: #2ecc71;
    color: white;
    grid-column: 2;
  }
  
  .btn-down {
    background: #e74c3c;
    color: white;
    grid-column: 2;
  }
  
  .btn-left {
    background: #3498db;
    color: white;
  }
  
  .btn-right {
    background: #9b59b6;
    color: white;
  }
  
  .btn-cont-up {
    background: #2ecc71;
    color: white;
  }
  
  .btn-cont-down {
    background: #e74c3c;
    color: white;
  }
  
  .btn-cont-left {
    background: #3498db;
    color: white;
  }
  
  .btn-cont-right {
    background: #9b59b6;
    color: white;
  }
  
  .btn-stop {
    background: #7f8c8d;
    color: white;
    grid-column: 1 / 4;
    margin-top: 10px;
  }
  
  .btn-loop {
    background: #f39c12;
    color: white;
    grid-column: 1 / 4;
  }
  
  .btn-up:hover { background: #27ae60; }
  .btn-down:hover { background: #c0392b; }
  .btn-left:hover { background: #2980b9; }
  .btn-right:hover { background: #8e44ad; }
  .btn-cont-up:hover { background: #27ae60; }
  .btn-cont-down:hover { background: #c0392b; }
  .btn-cont-left:hover { background: #2980b9; }
  .btn-cont-right:hover { background: #8e44ad; }
  .btn-stop:hover { background: #636e72; }
  .btn-loop:hover { background: #e67e22; }
</style>
</head>
<body>
  <div class="container">
    <h2>Kontrol Lengan Robot</h2>
    
    <div class="status-box">
      Status: <span id="status-text">IDLE</span>
    </div>
    
    <div class="control-section">
      <div class="section-title">Gerakan Atas-Bawah</div>
      <div class="btn-grid">
        <button class='btn btn-up' onclick="send('up45')">↑ 45°</button>
        <button class='btn btn-up' onclick="send('up90')">↑ 90°</button>
        <button class='btn btn-cont-up' onclick="send('cont_up')">⏫ Terus</button>
        
        <button class='btn btn-down' onclick="send('down45')">↓ 45°</button>
        <button class='btn btn-down' onclick="send('down90')">↓ 90°</button>
        <button class='btn btn-cont-down' onclick="send('cont_down')">⏬ Terus</button>
      </div>
    </div>
    
    <div class="control-section">
      <div class="section-title">Gerakan Kiri-Kanan</div>
      <div class="btn-grid">
        <button class='btn btn-left' onclick="send('left45')">← 45°</button>
        <button class='btn btn-left' onclick="send('left90')">← 90°</button>
        <button class='btn btn-cont-left' onclick="send('cont_left')">⏪ Terus</button>
        
        <button class='btn btn-right' onclick="send('right45')">→ 45°</button>
        <button class='btn btn-right' onclick="send('right90')">→ 90°</button>
        <button class='btn btn-cont-right' onclick="send('cont_right')">⏩ Terus</button>
      </div>
    </div>
    
    <div class="btn-grid">
      <button class='btn btn-stop' onclick="send('stop')">⏹ BERHENTI SEMUA</button>
      <button class='btn btn-loop' onclick="send('loop5')">
        ↕ Oscillate 5x (Atas-Bawah)
      </button>
    </div>
  </div>

<script>
function send(cmd){
  fetch('/cmd?motor=' + cmd)
    .then(() => console.log('Sent:', cmd));
}

setInterval(() => {
  fetch('/status')
    .then(res => res.text())
    .then(txt => {
      document.getElementById('status-text').innerText = txt;
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

  if (cmd == "up45")         queueUp += 1;
  else if (cmd == "down45")  queueDown += 1;
  else if (cmd == "up90")    queueUp += 2;
  else if (cmd == "down90")  queueDown += 2;
  else if (cmd == "left45")  queueLeft += 1;
  else if (cmd == "right45") queueRight += 1;
  else if (cmd == "left90")  queueLeft += 2;
  else if (cmd == "right90") queueRight += 2;
  else if (cmd == "cont_up") startContinuous(true, false);
  else if (cmd == "cont_down") startContinuous(false, false);
  else if (cmd == "cont_left") startContinuous(true, true);
  else if (cmd == "cont_right") startContinuous(false, true);
  else if (cmd == "stop") {
    queueUp = queueDown = queueLeft = queueRight = 0;
    stopAllMotors();
    loopCounter = 0;
    loopRunning = false;
    state = IDLE;
  }
  else if (cmd == "loop5") {
    loopRunning = true;
    loopCounter = 0;
    startStep(true, false); // mulai dengan naik (vertikal)
  }

  server.send(200, "text/plain", "OK");
}


void handleStatus() {
  if (loopRunning) currentStatus = "Oscillate 5x (Atas-Bawah)";
  else if (state == STEP_UP) currentStatus = "Naik 45°";
  else if (state == STEP_DOWN) currentStatus = "Turun 45°";
  else if (state == STEP_LEFT) currentStatus = "Kiri 45°";
  else if (state == STEP_RIGHT) currentStatus = "Kanan 45°";
  else if (state == CONT_UP) currentStatus = "Naik Terus";
  else if (state == CONT_DOWN) currentStatus = "Turun Terus";
  else if (state == CONT_LEFT) currentStatus = "Kiri Terus";
  else if (state == CONT_RIGHT) currentStatus = "Kanan Terus";
  else currentStatus = "IDLE";

  server.send(200, "text/plain", currentStatus);
}


void startStep(bool direction, bool horizontal) {
  if (horizontal) {
    // Motor 2 - Kiri/Kanan
    if (direction) { // Kiri
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);
      state = STEP_LEFT;
    } else { // Kanan
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, HIGH);
      state = STEP_RIGHT;
    }
  } else {
    // Motor 1 - Atas/Bawah
    if (direction) { // Naik
      digitalWrite(IN3, HIGH);
      digitalWrite(IN4, LOW);
      state = STEP_UP;
    } else { // Turun
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, HIGH);
      state = STEP_DOWN;
    }
  }
  tStart = millis();
}

void startContinuous(bool direction, bool horizontal) {
  queueUp = queueDown = queueLeft = queueRight = 0;
  loopRunning = false;
  
  if (horizontal) {
    // Motor 2 - Kiri/Kanan
    if (direction) { // Kiri
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);
      state = CONT_LEFT;
    } else { // Kanan
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, HIGH);
      state = CONT_RIGHT;
    }
  } else {
    // Motor 1 - Atas/Bawah
    if (direction) { // Naik
      digitalWrite(IN3, HIGH);
      digitalWrite(IN4, LOW);
      state = CONT_UP;
    } else { // Turun
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, HIGH);
      state = CONT_DOWN;
    }
  }
}

void stopAllMotors() {
  stopMotor1();
  stopMotor2();
}

void stopMotor1() {
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

void stopMotor2() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
}
