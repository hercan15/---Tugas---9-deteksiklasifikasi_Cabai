#include <WiFi.h>
#include <WebServer.h>


const char* ssid     = "bismillah";
const char* password = "bunga123";
WebServer server(80);  // Port HTTP


// KIRI
const int IN1 = 14;
const int IN2 = 12;
const int ENA = 13;
// KANAN
const int IN3 = 27;
const int IN4 = 26;
const int ENB = 25;

String lastCmd = "";  

void setup() {
  Serial.begin(115200);

  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT); pinMode(ENA, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT); pinMode(ENB, OUTPUT);

  digitalWrite(ENA, HIGH);  
  digitalWrite(ENB, HIGH);  

 /* ip dinamis */
  WiFi.begin(ssid, password);
  Serial.print("Menghubungkan WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(400); Serial.print(".");
  }

  Serial.println("\n✓ WiFi Terhubung!");
  Serial.print("Alamat IP: "); Serial.println(WiFi.localIP());

  server.on("/", handleRoot);
  server.on("/cmd", handleCmd);
  server.begin();
}

void loop() {
  server.handleClient();
}

void handleRoot() {
  String page = R"rawliteral(
<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>Kendali Motor DC</title>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<style>
  body{font-family:Arial;text-align:center;background:#e7f0fa;padding:40px}
  h2{margin-bottom:20px}
  .btn{width:160px;height:60px;font-size:18px;font-weight:bold;
       border:none;border-radius:12px;margin:10px;cursor:pointer}
  .green{background:#4caf50;color:white}
  .blue{background:#2196f3;color:white}
  .orange{background:#ff9800;color:white}
  .red{background:#f44336;color:white}
</style></head><body>
<h2>Kontrol 4 Arah</h2>

<button class='btn green'  onclick="send('mundur')">⬆️ MUNDUR</button><br>
<button class='btn orange' onclick="send('kiri')">⬅️ BEL0K KIRI</button>
<button class='btn orange' onclick="send('kanan')">BEL0K KANAN ➡️</button><br>
<button class='btn blue'   onclick="send('maju')">⬇️ MAJU</button><br>
<button class='btn red'    onclick="send('stop')">⏹️ STOP</button>

<script>
function send(cmd){
  fetch("/cmd?motor="+cmd).then(()=>console.log("Sent:",cmd));
}
</script></body></html>
)rawliteral";

  server.send(200, "text/html", page);
}

void handleCmd() {
  String cmd = server.arg("motor");
  Serial.println("Dapat perintah: " + cmd);

  if (cmd == "mundur")      gerakMundur();
  else if (cmd == "maju") gerakMaju();
  else if (cmd == "kiri")   belokKiri();
  else if (cmd == "kanan")  belokKanan();
  else if (cmd == "stop")   stopMotor();

  lastCmd = cmd;
  server.send(200, "text/plain", "OK");
}

void gerakMundur() {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);   
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);   
}

void gerakMaju() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);   // kiri mundur
  digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);   // kanan mundur
}

void belokKiri() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);   // kiri mundur
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);   // kanan maju
}

void belokKanan() {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);   // kiri maju
  digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);   // kanan mundur
}

void stopMotor() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
}
