# Belief-Guided Semantic Frontier Exploration for Cross-Floor Object Navigation

[![License](https://img.shields.io/badge/License-Academic%20Use-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![ROS](https://img.shields.io/badge/ROS-Noetic%2FMelodic-orange.svg)](https://www.ros.org/)

## ⚠️ Version Notice

> **Current Release**: March 2026 version (stable)  
> **Latest Code**: Undergoing reorganization — will be released upon paper acceptance

The current repository contains the **March 2026 stable version** of our codebase, which corresponds to the experimental results reported in the paper. A cleaner, better-documented version with additional features is under active reorganization and will be made publicly available after paper acceptance.

| Version | Status | Description |
|---------|--------|-------------|
| March 2026 | ✅ Stable | Codebase used for paper experiments |
| Latest | 🚧 Pending | Refactored code with improved documentation |

## 📖 Overview

This repository contains the implementation of our paper:

> **Belief-Guided Semantic Frontier Exploration for Cross-Floor Object Navigation**  

We propose a novel framework for **language-guided cross-floor object navigation** in multi-floor indoor environments. Our method enables robots to:

- Parse natural language instructions into structured tasks (floor intent, room category, target object)
- Build and maintain a **cross-floor semantic topological graph** online
- Switch adaptively between **object search**, **semantic frontier exploration**, and **cross-floor transfer**
- Register **stair connectors** as reusable topological links between floors

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **Task-State-Aware Frontier Utility** | Dynamically switches between stair search and object search based on task state |
| **Object-Room Knowledge Graph** | Updates region-level semantic belief using detected objects and prior probabilities |
| **Stair Connector Registration** | Converts stable stair/handrail/railing observations into reusable floor-transfer nodes |
| **Cross-Floor Topological Graph** | Maintains floor-relative state without requiring absolute floor numbers |

## 🏆 Results

### HM3D-v1 Benchmark

| Method | SR (%) | SPL (%) | CFSR (%) |
|--------|--------|---------|----------|
| VFLM (ICRA 2024) | 52.0 | 29.9 | 18.7 |
| ASCENT (RA-L 2026) | 64.2 | 31.6 | 47.4 |
| **Ours** | **73.9** | **33.8** | **55.1** |

### Real-World Experiments (40 episodes)

| Task | Success Rate | Avg. Path Length |
|------|--------------|------------------|
| Same-Floor | 85.0% | 32.5m |
| Cross-Floor | 75.0% | 70.7m |
| **Overall** | **80.0%** | 51.6m |

## 🏗️ System Architecture
