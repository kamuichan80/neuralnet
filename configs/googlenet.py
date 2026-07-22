#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""GoogLeNet config (Szegedy et al., 2014). Inception-stage channel tuples are architecturally
fixed (irregular pooling placement between stages) so they stay in-code rather than here."""

model = dict(
    type='GoogLeNet',
    num_classes=1000,
)
