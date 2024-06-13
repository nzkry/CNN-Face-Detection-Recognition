//f91e3a

/*Dev Log
1. Use HiveMq as MQTT broker
2. Simple IoT system to open/off solenoid door
3. WifiManager lib for wifi webclient config
4. 4/2/2023: Add delay for relay input HIGH (Door open), buzz alarm/send warning message to HiveMQ if relay input HIGH (door is still open) after delay 
5. 10/2/2023: Add buzzer, manual switch + 5 seconds delay, secure connection
6. Remove active buzzer, replace with passive and directly connect to relay = buzz on each time door open + off when door close
7. Add support for magnetic contact switch = indicate open/close door
8. Get nodeMCU UID to assign as door UID
9. Fix door logic = door always close and auto close after 10 seconds, add 2 UID to distinguished 2 doors ID-DCS= e101c2, , ID-DLIS=4a6c4d , 3 prototype: 8984b0
*/

#if defined(ESP32)
#include <WiFi.h>
#elif defined(ESP8266)
#include <ESP8266WiFi.h>
#endif
#include <WiFiManager.h>
#include <PubSubClient.h>

//MQTT Broker -for security make seperate file
const char* mqtt_server = "broker.hivemq.com";  // HiveMQ broker URL
const char* door5_status = "door5";
const char* door6_status = "door6";
const char* doorstat_topic = "doorstatus";  //old" magstatus
const char* door_uid3 = "dooruid3";
const char* face_detect = "FaceDetection";
const int mqtt_port = 8883;
const int relay1 = 5;
const int relay2 = 4;
const int buzzer = 14;
const int swtch = 12;
const int rstSwitch = 13;
const int deskSW = 2;
int manualSWCurrent, manualSWLast, deskSWCurrent, deskSWLast, rstStateCurrent, rstStateLast;
String doorUID;


WiFiClientSecure espClient;
PubSubClient client(espClient);

/****Functions****/
// Wifi Setup
void setupWifi() {
  //Init WifiManager
  WiFiManager wifiManager;
  Serial.print("\n\nConnecting Wifi: ");
  //wifiManager.resetSettings();
  wifiManager.autoConnect("Smart Door WiFi Setup");
  Serial.print("WiFi Status: Connected");
}

//HiveMQ connection setup
void reconnect() 
{
  //Connect HiveMQ server, loop until connected
  while (!client.connected()) 
  {
    String clientId = "ESP8266Client-";  // Create a random client ID - Change to unique ID
    clientId += doorUID;
    Serial.println("\nHiveMQ:" + clientId);
    // Attempt to connect
    if (client.connect(clientId.c_str(), mqtt_username, mqtt_password)) 
    {
      Serial.println("\nHiveMQ: Connected");
      client.subscribe(door5_status);
      client.subscribe(door6_status);
      client.subscribe(doorstat_topic);
      client.subscribe(door_uid3);
      client.subscribe(face_detect);

      publishMessage(doorstat_topic, String("Door Initialized"), true);

      if (doorUID == "f91e3a") 
      {
        publishMessage(door_uid3, String("Prototype ONLINE"), true);  //push nodeMCU chip UID
      } 
      else 
      {
        publishMessage(door_uid3, String("Prototype OFFLINE"), true);
      }

    } 
    else 
    {
      Serial.print("Connection failed: ");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

//MQTT callback
void callback(char* topic, byte* payload, unsigned int length) 
{
  String incommingMessage = "";
  for (int i = 0; i < length; i++) incommingMessage += (char)payload[i];
  Serial.println("HiveMQ Incomming Message[" + String(topic) + "]: " + incommingMessage);
  
  if (doorUID == "f91e3a") //Prototype
  {  
    //Check Door 5 Status
    if (strcmp(topic, door5_status) == 0) 
    { //Auto lock door after 10 seconds
     if (incommingMessage.equals("Door Left Switch")) 
      { //relay HIGH = OPEN, LOW = CLOSE
        publishMessage(doorstat_topic, String("Left Switch Triggered, Door Open"), true);
        pinMode(relay1, HIGH);
        digitalWrite(buzzer, HIGH);
        delay(3000);
        digitalWrite(buzzer, LOW);
        delay(7000);
        publishMessage(door5_status, String("Door Left Close"), true);
      } 
      else 
      {
        pinMode(relay1, LOW);  //close door
        publishMessage(doorstat_topic, String("Door Left: CLOSED"), true);
      }
    }
    
    //Check Door 6 Status
    if (strcmp(topic, door6_status) == 0) 
    {
      if (incommingMessage.equals("Door Right Switch")) 
      {
        publishMessage(doorstat_topic, String("Right Switch Triggered, Door Open"), true);
        pinMode(relay2, HIGH);
        digitalWrite(buzzer, HIGH);
        delay(3000);
        digitalWrite(buzzer, LOW);
        delay(7000);
        publishMessage(door6_status, String("Door Right Close"), true);
      } 
      else 
      {
        pinMode(relay2, LOW);  //close door
        publishMessage(doorstat_topic, String("Door Right: CLOSED"), true);
      }
    }
    //Check Face Detection
    if (strcmp(topic, face_detect) == 0) 
    { //Auto lock door after 10 seconds
     if (incommingMessage.equals("Face Detected")) 
      { //relay HIGH = OPEN, LOW = CLOSE
        publishMessage(doorstat_topic, String("Face Detected, Door Open"), true);
        pinMode(relay1, HIGH);
        pinMode(relay2, HIGH);
        digitalWrite(buzzer, HIGH);
        delay(3000);
        digitalWrite(buzzer, LOW);
        delay(7000);
        publishMessage(face_detect, String("Door Close"), true);
      } 
      else 
      {
        pinMode(relay1, LOW);  //close door
        pinMode(relay2, LOW);
        publishMessage(doorstat_topic, String("Door: CLOSED"), true);
      }
    }
  }
}

//MQTT publising as string
void publishMessage(const char* topic, String payload, boolean retained) 
{
  if (client.publish(topic, payload.c_str(), true))
    Serial.println("Message publised [" + String(topic) + "]: " + payload);
}

void onManualSwitch()
{
 //Open door
    pinMode(relay1, HIGH);
    pinMode(relay2, HIGH);
    digitalWrite(buzzer, HIGH);
    Serial.println("Manual Switch ON: Door Open");
    publishMessage(doorstat_topic, String("Manual Switch Triggered, Door Open"), true);
    delay(3000);
    digitalWrite(buzzer, LOW);

    //Close door after 10 seconds
    delay(10000);
    pinMode(relay1, LOW);
    pinMode(relay2, LOW);
    Serial.println("Manual Switch OFF: Door Close");
    publishMessage(doorstat_topic, String("Door: CLOSED"), true);
}

/*****Setup && Loop*****/
void setup() 
{
  Serial.begin(9600);
  while (!Serial) delay(1);
  doorUID = String(ESP.getChipId(), HEX);
  Serial.printf("\n ESP8266 Chip id = %08X\n", doorUID);  //get nodeMCU UID
  pinMode(relay1, OUTPUT);
  pinMode(relay2, OUTPUT);
  pinMode(buzzer, OUTPUT);
  pinMode(swtch, INPUT_PULLUP);
  pinMode(rstSwitch, INPUT_PULLUP);
  pinMode(deskSW, INPUT_PULLUP);
  setupWifi();
  //WiFi Client Init
#ifdef ESP8266
  espClient.setInsecure();
#else
  espClient.setCACert(root_ca);
#endif
  //MQTT Init
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
}

void loop() {
  pinMode(relay1, LOW);
  pinMode(relay2, LOW);

  //init reset button to restart nodemcu
  rstStateLast = rstStateCurrent;
  rstStateCurrent = digitalRead(rstSwitch);
  if (rstStateLast == HIGH && rstStateCurrent == LOW) {
    Serial.printf("Restart NodeMCU..");
    ESP.restart();
  }
  //Front Desk Switch: Open door for 10 seconds.
  deskSWLast = deskSWCurrent;
  deskSWCurrent = digitalRead(deskSW);
  if (deskSWLast == HIGH && deskSWCurrent == LOW) {
   onManualSwitch();
  }
  //Manual Switch: Open door for 10 seconds.
  manualSWLast = manualSWCurrent;
  manualSWCurrent = digitalRead(swtch);
  if (manualSWLast == HIGH && manualSWCurrent == LOW) {
    onManualSwitch();
  }
  //Client listener
  if (!client.connected()) reconnect();
  client.loop();
}
