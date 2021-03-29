int leftMuscle = 0;
int rightMuscle = 0;
bool leftSafety = true;
bool rightSafety = true;

void setup() {
  Serial.begin(9600);

  //Safety Button Inputs
  pinMode(2, INPUT);
  pinMode(3, INPUT);
  attachInterrupt(digitalPinToInterrupt(2), updateSafety, CHANGE);
  attachInterrupt(digitalPinToInterrupt(3), updateSafety, CHANGE);

  //Muscle Sensor inputs
  // A0 A1
}

void updateSafety() {
  leftSafety = digitalRead(2);
  rightSafety = digitalRead(3);
  if (leftSafety && rightSafety) Serial.println("Safety OFF");
  else Serial.println("Safety ON");
}

void loop() { 
  delay(1000);
  
  leftMuscle = analogRead(A0);
  rightMuscle = analogRead(A1);
  
  Serial.print("Left:\t");Serial.print(leftMuscle);
  Serial.print("\tRight:\t");Serial.print(rightMuscle);
  Serial.print("\n");
}
