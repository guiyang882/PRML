/* Variables */
var x1 >= 0;
var x2 >= 0;
var x3 >= 0;

/* Object function */
maximize z: 3*x1 + 2*x2;

/* Constrains */
s.t. con1: 2*x1 + x2 <= 100;
s.t. con2: x2 + x1  <= 80;
s.t. con3: x1 <= 40;

end;
