OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
cx q[0],q[2];
cx q[1],q[2];
rz(0.7853981633974484) q[2];
cx q[1],q[2];
cx q[0],q[2];