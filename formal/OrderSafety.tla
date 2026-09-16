---- MODULE OrderSafety ----
EXTENDS Naturals

CONSTANT MaxQty
VARIABLES state, halted, submitted, filled, writer

vars == <<state, halted, submitted, filled, writer>>
States == {"NEW", "RESERVED", "SUBMITTED", "ACKNOWLEDGED", "PARTIAL", "FILLED", "CANCELED", "REJECTED"}

Init ==
    /\ state = "NEW"
    /\ halted = FALSE
    /\ submitted = 0
    /\ filled = 0
    /\ writer = "ledger"

Reserve == /\ state = "NEW" /\ ~halted /\ state' = "RESERVED"
           /\ UNCHANGED <<halted, submitted, filled, writer>>
Submit == /\ state = "RESERVED" /\ ~halted /\ submitted = 0
          /\ state' = "SUBMITTED" /\ submitted' = 1
          /\ UNCHANGED <<halted, filled, writer>>
Ack == /\ state = "SUBMITTED" /\ state' = "ACKNOWLEDGED"
       /\ UNCHANGED <<halted, submitted, filled, writer>>
Fill == /\ state \in {"ACKNOWLEDGED", "PARTIAL"} /\ filled < MaxQty
        /\ filled' = filled + 1
        /\ state' = IF filled' = MaxQty THEN "FILLED" ELSE "PARTIAL"
        /\ UNCHANGED <<halted, submitted, writer>>
Cancel == /\ state \in {"RESERVED", "SUBMITTED", "ACKNOWLEDGED", "PARTIAL"}
          /\ state' = "CANCELED"
          /\ UNCHANGED <<halted, submitted, filled, writer>>
Halt == /\ halted' = TRUE /\ UNCHANGED <<state, submitted, filled, writer>>
Resume == /\ halted' = FALSE /\ UNCHANGED <<state, submitted, filled, writer>>

Next == Reserve \/ Submit \/ Ack \/ Fill \/ Cancel \/ Halt \/ Resume
TypeInvariant == state \in States /\ halted \in BOOLEAN /\ submitted \in 0..1 /\ filled \in 0..MaxQty
NoOverfill == filled <= MaxQty
SingleSubmit == submitted <= 1
SingleWriter == writer = "ledger"
Spec == Init /\ [][Next]_vars

====
