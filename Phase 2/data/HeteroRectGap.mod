NEURON {
  THREADSAFE
  POINT_PROCESS HeteroRectGap
  NONSPECIFIC_CURRENT i
  RANGE i, g, gmax_open, gmax_closed, use_transfer, vgap_xfer, orientation, vhalf, vslope
  POINTER vgap_ptr
}

PARAMETER {
  gmax_open = 0 (nanosiemens)
  gmax_closed = 0 (nanosiemens)
  orientation = 1
  vhalf = 0 (millivolt)
  vslope = 5 (millivolt)
  use_transfer = 0
}

ASSIGNED {
  v (millivolt)
  i (nanoamp)
  vgap_ptr (millivolt)
  vgap_xfer (millivolt)
  g (nanosiemens)
  vpeer (millivolt)
  vj_oriented (millivolt)
  gate
}

BREAKPOINT {
  if (use_transfer > 0.5) {
    vpeer = vgap_xfer
  } else {
    vpeer = vgap_ptr
  }
  vj_oriented = orientation * (v - vpeer)
  gate = 1.0 / (1.0 + exp(-(vj_oriented - vhalf) / vslope))
  g = gmax_closed + (gmax_open - gmax_closed) * gate
  i = (v - vpeer) * g * 0.001
}
