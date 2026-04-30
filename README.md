# TFM: Detección de Defectos mediante CNN Custom y Arquitecturas Avanzadas

Este repositorio contiene el desarrollo integral de un sistema de visión artificial para la detección de defectos industriales. El proyecto evoluciona desde clasificadores binarios básicos hasta un **detector de defectos implementado a medida**, integrando técnicas de estimación de incertidumbre y explicabilidad.

## 🛠️ El Núcleo del Proyecto: Detector de Defectos "A Mano"
El componente más relevante de esta investigación se encuentra en el notebook **`06_detector_defectos_mejorado.ipynb`** (rama `exp_estimacion_incertidumbre` / `exp_explicabilidad`). A diferencia de los clasificadores estándar, este módulo implementa:

- **Arquitectura de Detección Personalizada:** Un diseño de red que no solo clasifica la imagen global, sino que está optimizado para localizar anomalías.
- **Implementación "desde cero":** Se han definido manualmente las capas de extracción de características y las cabezas de detección para adaptarse a la morfología específica de los defectos del dataset.
- **Refinamiento de Predicciones:** Uso de técnicas para mejorar la precisión en la localización, superando las limitaciones de las arquitecturas de clasificación tradicionales aplicadas a detección.
- **Integración de Incertidumbre:** El detector es capaz de informar sobre la confianza de la localización, permitiendo un filtrado más robusto de falsos positivos en entornos industriales.

---

## 🚀 Resumen de Ramas y Evolución

### 📂 Clasificación y Base (`master`)
Establece los cimientos del proyecto mediante la comparación de diferentes enfoques de aprendizaje:
- **Custom CNN:** Entrenamiento de una red sencilla desde cero para establecer una línea base (baseline).
- **Transfer Learning (ResNet18):** Uso de pesos pre-entrenados para acelerar la convergencia.
- **Fine-Tuning (EfficientNet-B2):** Ajuste fino de una de las arquitecturas más eficientes del estado del arte para maximizar el F1-Score.

### 📂 Detección y Refinamiento (`exp_detector_defectos_...`)
En estas ramas se realiza la transición de "clasificar una imagen" a "localizar un defecto":
- Implementación de algoritmos de detección específicos.
- Iteraciones de refinamiento donde se ajustan funciones de pérdida (Loss Functions) personalizadas.

### 📂 Fiabilidad y Explicabilidad (`exp_estimacion_incertidumbre` / `exp_explicabilidad`)
Ramas críticas para la validación del modelo en el mundo real:
- **Incertidumbre:** Implementación de mecanismos para medir la duda del modelo, esencial para la seguridad en procesos de fabricación.
- **XAI (Explicabilidad):** Uso de mapas de calor para verificar que el detector "mira" realmente el defecto y no artefactos del fondo de la imagen.

### 📂 Metodología (`exp_DCRISP`)
Organización del flujo de trabajo bajo la metodología **D-CRISP**, asegurando la trazabilidad desde la comprensión de los datos hasta el despliegue del modelo.

---

## 📋 Requisitos y Configuración
El proyecto utiliza un entorno basado en **PyTorch** y **Jupyter**.
1. Clonar el repositorio.
2. Seguir el orden numérico de los notebooks para reproducir los experimentos:
   - `01_preparar_datos`: Limpieza y aumento de datos (Augmentation).
   - `02` a `04`: Entrenamiento de clasificadores.
   - `05` y `06`: Implementación y mejora del detector manual.
