#include "CmdMessenger.h"
#include <Servo.h>

Servo myservo1;
Servo myservo2;


enum {
    green_color,
    greeen,
    red_color,
    reed,
    what_camera,
    state_of_camera,
    change_var,
    var_news,
    error,
};

const int BAUD_RATE = 9600;
CmdMessenger c = CmdMessenger(Serial,',',';','/');

const int buttonPin = A0; 

int buttonVarstate = 0;

const int ledPin = 13;

int position1=0;
int position2=0;

int redcolor=0;
int mainmotor=1;
int makecomand=0;
int checking =0;
int lightmeasure=0;
int steps =0;
void on_what_camera(void){   
    if(checking==1) {  
    buttonVarstate = 1;
    checking = 0; 
    }
    else {buttonVarstate = 0;}
    c.sendCmd(state_of_camera, buttonVarstate);
}



void on_change_var(void){
    buttonVarstate = 0;
    c.sendCmd(var_news, buttonVarstate);
    
}

void on_green_color(void){
 // digitalWrite(green, HIGH);
 // digitalWrite(red, LOW);
 makecomand=1;   
 c.sendCmd(greeen, ledPin);
}

void on_red_color(void){
 redcolor=1;
 makecomand=1;  
 c.sendCmd(reed, ledPin);
}

/* callback */
void on_unknown_command(void){
    c.sendCmd(error,"Command without callback.");
}

/* Attach callbacks for CmdMessenger commands */
void attach_callbacks(void) { 
    c.attach(green_color,on_green_color);
    c.attach(red_color,on_red_color);
    c.attach(what_camera,on_what_camera);
    c.attach(change_var,on_change_var);
    c.attach(on_unknown_command);
}

void setup() {
    myservo1.attach(8);
    myservo2.attach(13);
    
    Serial.begin(BAUD_RATE);
    pinMode(ledPin, OUTPUT);
    pinMode(buttonPin, INPUT);
    attach_callbacks();
    Serial.begin(9600);
}

void loop() {
    //c.feedinSerialData();
    lightmeasure=analogRead(buttonPin);
    
    if(255-lightmeasure*3<10)
      { 
       mainmotor = 0;
       checking = 1;
        
       myservo1.detach();
       while(makecomand==0)
       {
        c.feedinSerialData();
       }
       
      
       if(redcolor==1){
         
           for(position2 = 90; position2 > 0; position2 -= 1)  // goes from 0 degrees to 180 degrees 
              {                                  // in steps of 1 degree 
                myservo2.write(position2);              // tell servo to go to position in variable 'pos' 
                delay(15);                       // waits 15ms for the servo to reach the position 
              }
       
       redcolor=0; 

       }
       
       else {
            for(position2 = 90; position2 > 0; position2 -= 1)  // goes from 0 degrees to 180 degrees 
              {                                  // in steps of 1 degree 
                myservo2.write(position2);              // tell servo to go to position in variable 'pos' 
                delay(15);                       // waits 15ms for the servo to reach the position 
              } 
       }
       for(position2 = 90; position2 > 0; position2 -= 1)  // goes from 0 degrees to 180 degrees 
              {                                  // in steps of 1 degree 
                myservo2.write(position2);              // tell servo to go to position in variable 'pos' 
                delay(15);                       // waits 15ms for the servo to reach the position 
              } 
       
       makecomand=0;
      }
    else if(mainmotor==0)
    {
      mainmotor=1;
      myservo1.attach(8);
    }
    
    if(position1>=180) {position1=0; } // goes from 0 degrees to 180 degrees 
    position1 += 1;                                // in steps of 1 degree 
    myservo1.write(position1);              // tell servo to go to position in variable 'pos' 
    delay(2); 
    
  }
    /*
    for(steps=0; steps<5; steps++)
       {
     if(position1 == 180) {position1=0; } // goes from 0 degrees to 180 degrees 
   position1 += 10;                                // in steps of 1 degree 
    myservo1.write(position1);              // tell servo to go to position in variable 'pos' 
    delay(30);
       }
    
    
    */

