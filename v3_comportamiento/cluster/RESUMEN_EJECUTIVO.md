# Resumen Ejecutivo - Metodologías de Clasificación de Comportamiento Bovino

## Contexto del Proyecto

Este proyecto implementa dos metodologías complementarias para la clasificación automatizada de comportamientos en ganado bovino mediante técnicas de aprendizaje profundo y visión por computadora.

---

## Metodologías Implementadas

### 🔵 **Metodología 1: Aprendizaje No Supervisado con Clustering**

**Objetivo:** Identificar patrones comportamentales sin etiquetas previas

**Pipeline:**
```
Video → VGG16 → LSTM → Features (512-dim) → K-Means/DBSCAN → Clusters → Validación Manual → Labels
```

**Características técnicas:**
- **Extractor de features:** VGG16 preentrenado (ImageNet) + LSTM bidireccional
- **Clustering:** K-Means (k=4) y DBSCAN para outliers
- **Visualización:** t-SNE y PCA para validación visual
- **Detección de anomalías:** Isolation Forest sobre patrones temporales
- **Output:** Pseudo-etiquetas validadas manualmente

**Ventajas:**
- ✅ No requiere etiquetas iniciales
- ✅ Descubre patrones automáticamente
- ✅ Detección de anomalías comportamentales
- ✅ Alta interpretabilidad (visualizaciones)

**Resultados:**
- Embeddings de 512 dimensiones por secuencia
- Identificación de 4 clusters principales
- Detección de outliers (~10% del dataset)
- Métricas: Silhouette Score, Davies-Bouldin Index, ARI

---

### 🟢 **Metodología 2: Aprendizaje Supervisado (Dual)**

**Objetivo:** Clasificación directa usando etiquetas validadas

#### **Variante A: CNN-LSTM (Secuencias Temporales)**

**Pipeline:**
```
Secuencia (16 frames) → VGG16 → Feature Reducer → LSTM → Clasificador → 4 clases
```

**Especificaciones:**
- **Input:** Secuencias de 16 frames (224×224 RGB)
- **Arquitectura:** VGG16 (10 capas congeladas) + LSTM (2 capas, 512 hidden)
- **Parámetros:** 31.9M total (99.2% entrenables)
- **Dataset:** ~500 secuencias
- **Accuracy esperado:** 80-90%

**Ventajas:**
- ✅ Captura dependencias temporales
- ✅ Robusto ante frames individuales ruidosos
- ✅ Mejor para comportamientos dinámicos

**Limitaciones:**
- ⚠️ Menor cantidad de muestras
- ⚠️ Mayor consumo de memoria (6-8 GB)
- ⚠️ Entrenamiento más lento

#### **Variante B: CNN Simple (Frames Individuales)**

**Pipeline:**
```
Imagen → VGG16 → Clasificador profundo → 4 clases + Features (4096-dim)
```

**Especificaciones:**
- **Input:** Imágenes individuales (224×224 RGB)
- **Arquitectura:** VGG16 + Clasificador denso (4096→1024→4)
- **Dataset:** ~10,000 imágenes
- **Accuracy esperado:** 75-85%

**Ventajas clave:**
- ✅ Dataset 20× más grande
- ✅ Entrenamiento 4× más rápido
- ✅ Extracción directa de features (4096-dim)
- ✅ Ideal para clustering posterior

**Aplicaciones post-entrenamiento:**
1. **Clustering sobre embeddings:** Validación del aprendizaje
2. **Búsqueda de similitud:** Encontrar imágenes parecidas
3. **Detección de casos ambiguos:** Identificar predicciones dudosas
4. **Transfer learning:** Usar features para otras tareas

---

## Flujo de Trabajo Integrado

```
┌─────────────────────────────────────────────────────────┐
│ FASE 1: EXPLORACIÓN NO SUPERVISADA                      │
├─────────────────────────────────────────────────────────┤
│ 1. Extracción features (VGG16+LSTM)                     │
│ 2. Clustering K-Means (k=4)                             │
│ 3. Visualización t-SNE/PCA                              │
│ 4. Validación visual manual                             │
│ 5. Generación de labels validados                       │
│ 6. Detección de anomalías (Isolation Forest)            │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ FASE 2: ENTRENAMIENTO SUPERVISADO                       │
├─────────────────────────────────────────────────────────┤
│ 7. Entrenamiento CNN-LSTM (secuencias)                  │
│ 8. Evaluación temporal                                  │
│ 9. Análisis de overfitting                              │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ FASE 3: EXTRACCIÓN Y ANÁLISIS DE FEATURES               │
├─────────────────────────────────────────────────────────┤
│ 10. Entrenamiento CNN simple (frames)                   │
│ 11. Extracción embeddings (4096-dim)                    │
│ 12. Clustering sobre embeddings                         │
│ 13. Búsqueda de similitud                               │
│ 14. Identificación casos ambiguos                       │
└─────────────────────────────────────────────────────────┘
```

---

## Métricas de Evaluación

### Clustering (No Supervisado)
- **Silhouette Score:** Calidad de separación de clusters (1.0 = perfecto)
- **Davies-Bouldin Index:** Compacidad de clusters (0 = mejor)
- **ARI (Adjusted Rand Index):** Similitud con labels manuales

### Clasificación (Supervisado)
- **Accuracy:** Exactitud global
- **Precision/Recall/F1:** Por cada clase de comportamiento
- **Confusion Matrix:** Análisis de errores
- **Overfitting Gap:** Train Acc - Val Acc (<10% = bueno)

---

## Resultados Comparativos

| Aspecto | No Supervisado | CNN-LSTM | CNN Simple |
|---------|---------------|----------|------------|
| **Requiere labels** | ❌ No | ✅ Sí | ✅ Sí |
| **Tamaño dataset** | Variable | ~500 seq | ~10K imgs |
| **Info temporal** | ✅ Sí | ✅ Sí | ❌ No |
| **Accuracy** | N/A | 80-90% | 75-85% |
| **Velocidad** | Media | Lenta | Rápida |
| **GPU Memory** | 4-6 GB | 6-8 GB | 2-3 GB |
| **Detecta anomalías** | ✅ Sí | ❌ No | ❌ No |
| **Features dim** | 512 | - | 4096 |
| **Interpretabilidad** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |

---

## Herramientas y Scripts

### Scripts Principales

| Script | Función | Metodología |
|--------|---------|------------|
| `main.py` | Pipeline no supervisado completo | No Supervisada |
| `train_supervised.py` | Entrenamiento CNN-LSTM | Supervisada A |
| `train_cnn_simple.py` | Entrenamiento CNN simple | Supervisada B |
| `extract_features.py` | Extracción embeddings | Post-procesamiento |
| `use_features_clustering.py` | Clustering sobre features | Análisis |
| `use_features_similarity.py` | Búsqueda de similitud | Análisis |

### Arquitectura de Archivos

```
v3_comportamiento/cluster/
├── config.py                      # Configuración central
├── models.py                      # Arquitecturas CNN-LSTM
├── dataset.py                     # Loaders de datos
├── anomaly_detection.py           # Isolation Forest
├── utils.py                       # Utilidades
│
├── main.py                        # 🔵 Pipeline no supervisado
├── train_supervised.py            # 🟢 Entrenamiento CNN-LSTM
├── train_cnn_simple.py            # 🟢 Entrenamiento CNN simple
│
├── extract_features.py            # Extracción embeddings
├── use_features_clustering.py     # Análisis clustering
├── use_features_similarity.py     # Búsqueda similitud
│
├── METODOLOGIA_COMPLETA.md        # 📄 Documento técnico completo
├── GUIA_USO_FEATURES.md           # 📄 Guía de uso
└── README_CNN_SIMPLE.md           # 📄 Doc CNN simple
```

---

## Outputs Generados

### Metodología No Supervisada
```
resultados_comportamiento/cluster_/
├── embeddings.npy                 # Features 512-dim
├── pseudo_labels_unsupervised.json
├── behavior_patterns.json         # Patrones emergentes
├── anomalies_detected.csv         # Outliers
├── unsupervised_report.txt        # Reporte completo
└── visualizaciones/
    ├── clusters_tsne.html
    ├── clusters_pca.html
    └── distribution.png
```

### Metodología Supervisada
```
resultados_comportamiento/modelos_entrenados/
├── best_supervised_model.pth      # CNN-LSTM entrenado
├── best_cnn_simple_model.pth      # CNN simple entrenado
├── training_metrics.png           # Curvas entrenamiento
└── cnn_training_metrics.png

resultados_comportamiento/features_extraidas/
├── embeddings.npy                 # Features 4096-dim
├── embeddings.predictions.json    # Predicciones
├── embeddings.pkl                 # Todo junto
├── features_tsne_visualization.png
├── optimal_k.png                  # Análisis k óptimo
├── clusters_tsne.png              # Clustering
├── top10_[comportamiento].png     # Mejores ejemplos
└── ambiguous_cases.png            # Casos dudosos
```

---

## Innovaciones Clave

1. **Pipeline híbrido:** Combina aprendizaje no supervisado (exploración) con supervisado (refinamiento)

2. **Doble extracción de features:** 
   - 512-dim (LSTM) para análisis temporal
   - 4096-dim (CNN) para similitud y clustering

3. **Validación iterativa:** Clusters automáticos → validación manual → entrenamiento supervisado

4. **Detección multi-nivel de anomalías:**
   - Clustering (outliers)
   - Isolation Forest (patrones temporales)
   - Casos ambiguos (baja confianza)

5. **Transfer learning optimizado:** Solo 10 capas VGG16 congeladas (balance adaptación/velocidad)

---

## Conclusiones

### ✅ Fortalezas del Sistema

- **Flexibilidad:** Múltiples enfoques según disponibilidad de datos
- **Escalabilidad:** Procesa miles de videos eficientemente
- **Robustez:** Detección automática de anomalías
- **Interpretabilidad:** Visualizaciones ricas en información
- **Reutilización:** Features extraídos útiles para múltiples tareas

### 📊 Casos de Uso

| Objetivo | Metodología Recomendada |
|----------|------------------------|
| Exploración inicial sin labels | No Supervisada |
| Máxima precisión temporal | CNN-LSTM Supervisada |
| Clustering y similitud | CNN Simple + Features |
| Detección de anomalías | No Supervisada + Isolation Forest |
| Sistema de producción | CNN Simple (rápido) |

### 🎯 Recomendación General

**Para proyectos nuevos:**
1. Iniciar con metodología no supervisada (exploración)
2. Validar clusters manualmente (generar labels)
3. Entrenar CNN simple (dataset grande, rápido)
4. Usar CNN-LSTM solo si necesitas análisis temporal

**Para sistemas en producción:**
- Usar CNN simple para inferencia rápida
- Ejecutar clustering periódico para detectar drift
- Monitorear casos ambiguos para mejora continua

---

**Elaborado por:** Sistema de Clasificación de Comportamiento Bovino  
**Proyecto:** Agrosavia  
**Fecha:** Octubre 2025
