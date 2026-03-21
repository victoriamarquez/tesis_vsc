# Pendientes — Mejoras al pipeline de modificación emocional

> Última actualización: marzo 2026
> Contexto: el pipeline funciona end-to-end, pero los resultados visuales no son buenos todavía.
> Dos líneas de trabajo para mejorarlos: (1) probar con imágenes de CelebA más limpias y
> correctamente alineadas; (2) aumentar el dataset BU-3DFE para obtener mejores vectores emocionales.

---

## Bloque 1 — Experimento CelebA: cara alineada con fondo negro

**Objetivo:** Ver si la modificación emocional funciona mejor cuando la imagen de CelebA se
procesa igual que las imágenes de BU-3DFE: pasando por el alineador de StyleGAN y con fondo negro.

**Hallazgo clave (del código):** Actualmente `celeba_processing.py` usa `img_align_celeba`
(el recorte propio de CelebA, 178×218 px) y va directo a la proyección, **sin pasar por
`align_images.py` de StyleGAN**. En cambio, el pipeline de BU-3DFE sí alinea las imágenes
con ese script, generando imágenes 1024×1024 en el formato FFHQ que el modelo espera.
Esa diferencia de preprocesamiento puede estar afectando la calidad del resultado en CelebA.
Los dos cambios a probar son: (a) pasar CelebA por el alineador de StyleGAN, y
(b) reemplazar el fondo con negro (igual que BU-3DFE).

### 1.1 Selección de imágenes de CelebA

> Esta parte la hacés vos a mano. El challenge es que las imágenes están en la computadora
> de la facultad, accesible solo por SSH con conexión inestable. La estrategia recomendada
> es pre-filtrar con código para reducir el conjunto a una cantidad manejable, y recién
> ahí revisar manualmente.

- [x] **[código]** Escribir un script que lea `list_attr_celeba.txt` y filtre imágenes
  candidatas descartando: `Blurry=1`, `Wearing_Hat=1`, `Eyeglasses=1`, `Smiling=1`.
  → Implementado en `proyecto_vsc/src/filter_celeba_candidates.py`. Genera copia de
  las imágenes seleccionadas + galería HTML para revisión visual + `candidatas.txt`.
- [x] **[ejecución]** Correr `filter_celeba_candidates.py` con `--n 100` para obtener
  candidatas suficientes para revisión manual.
- [x] **[manual]** Copiar la carpeta de salida a la máquina local con `scp` y revisar
  la galería HTML.
- [x] **[manual]** Seleccionar subconjunto final → **50 imágenes** guardadas en
  `images/CelebA/candidatas_revision/candidatas.txt`. Las 50 imágenes no seleccionadas
  fueron eliminadas de la carpeta.

### 1.2 Preparación de las imágenes

- [ ] **[código]** Modificar el pipeline de CelebA para que las imágenes seleccionadas pasen
  por `align_images.py` de StyleGAN (igual que BU-3DFE) antes de ser proyectadas. Validar
  que las imágenes alineadas resultantes se vean bien (1024×1024, cara centrada).
- [ ] **[código]** Implementar el reemplazo del fondo por negro sólido sobre las imágenes ya
  alineadas. El fondo de las imágenes de CelebA alineadas no es uniforme, así que esto
  requiere una máscara. Opciones: (a) usar un umbral simple si el fondo queda bastante
  uniforme después de alinear, o (b) usar segmentación con un modelo liviano como
  `rembg` o similar.
- [ ] **[manual]** Verificar visualmente un lote de imágenes preparadas (alineadas + fondo
  negro) antes de seguir con la proyección.

### 1.3 Ejecución del experimento

- [ ] **[ejecución]** Proyectar las imágenes preparadas al espacio latente
  (`project_selected_celeba_images_from_df` o el flujo actualizado).
- [ ] **[ejecución]** Aplicar los vectores emocionales sobre los NPZ generados
  (`process_emotions_celeba`), con todas las emociones y sus multiplicadores por defecto.
- [ ] **[manual]** Guardar los resultados en una carpeta identificada
  (ej: `images/CelebA/experimento_fondo_negro/`).
- [ ] **[manual]** Revisar visualmente los resultados para al menos 5 imágenes,
  comparando con los resultados anteriores (sin alinear, con fondo original).

### 1.4 Reporte al director

- [ ] **[manual]** Armar una selección representativa: para cada imagen de prueba, mostrar
  la imagen original de CelebA → imagen alineada con fondo negro → imagen modificada por
  emoción (una grilla o comparativa).
- [ ] **[mail]** Enviar imágenes de entrada preparadas + resultados de modificación emocional
  al director de tesis.

---

## Bloque 2 — Data Augmentation en BU-3DFE

**Objetivo:** Aumentar el número de imágenes del dataset original para calcular vectores
emocionales más robustos. Luego re-correr el pipeline completo y comparar los nuevos
vectores con los actuales.

**Nota sobre el dataset:** Las imágenes de BU-3DFE ya tienen fondo negro uniforme,
lo que simplifica el augmentation de fondo (no hay que segmentar la cara).

### 2.1 Entender el estado actual del dataset

- [ ] **[manual]** Confirmar cuántas imágenes tiene el dataset BU-3DFE actualmente
  (total y desglosado por emoción e intensidad).
- [ ] **[manual]** Confirmar si las augmentations se aplican sobre las imágenes originales
  `.bmp` (antes de alinear) o sobre las ya alineadas `.png`. Lo más limpio sería aplicarlas
  antes de alinear y luego alinear todo junto.

### 2.2 Implementar las transformaciones de augmentation (código)

- [ ] **[código]** Implementar **flip horizontal**: invertir la imagen en el eje X y
  guardarla con sufijo `_flip` en el nombre. (Nota: discutir con el director si el
  flip tiene sentido dado que la asimetría facial podría interferir con las emociones.)
- [ ] **[código]** Implementar **ruido gaussiano sobre la imagen completa**: agregar ruido
  con media 0 y desvío estándar configurable (experimentar con valores como σ=5, σ=10, σ=20).
- [ ] **[código]** Implementar **ruido solo en el fondo**: dado que el fondo ya es negro
  uniforme, se puede crear una máscara simple por umbral (píxeles cercanos a negro) y
  aplicar ruido gaussiano solo en esa región, sin tocar la cara.
- [ ] **[código]** Integrar estas tres transformaciones como funciones en un nuevo archivo
  `augmentation.py` dentro de `proyecto_vsc/src/`.
- [ ] **[código]** Escribir un script que aplique todas las transformaciones al dataset
  completo y guarde las imágenes augmentadas en una carpeta separada con la misma
  estructura de subdirectorios por persona que el dataset original
  (ej: `images/BU3DFE_augmented/`). Nombrar las imágenes de manera que sean compatibles
  con `parse_file_name` o documentar el cambio de convención.

### 2.3 Generar el dataset aumentado

- [ ] **[ejecución]** Correr el script de augmentation sobre todo el dataset BU-3DFE.
- [ ] **[manual]** Verificar visualmente algunos ejemplos de cada transformación
  (flip, ruido imagen, ruido fondo) para al menos 3 personas del dataset.
- [ ] **[manual]** Confirmar que la estructura de carpetas y los nombres de archivo son
  compatibles con lo que espera el pipeline.
- [ ] **[manual]** Anotar el total de imágenes del dataset aumentado.

### 2.4 Re-correr el pipeline completo con el dataset aumentado

- [ ] **[ejecución]** Correr la alineación sobre las imágenes nuevas
  (`align_all_images_from_df` con el nuevo dataset).
- [ ] **[ejecución]** Correr la proyección al espacio latente sobre todas las imágenes
  nuevas (`calculate_vectors` con flag `process`). ⚠️ Esto puede tardar mucho —
  planificar con tiempo.
- [ ] **[ejecución]** Calcular los nuevos vectores emocionales con el dataset aumentado
  (`python argument_parsing.py calculate_vectors`).
- [ ] **[manual]** Guardar los nuevos vectores en un archivo separado para no sobreescribir
  los actuales (ej: `datos/directions_regression_augmented.csv`).

### 2.5 Comparar los vectores nuevos con los originales

- [ ] **[ejecución]** Usar `method_comparison.py` para comparar los vectores originales vs.
  los nuevos por similitud coseno.
- [ ] **[ejecución]** Generar imágenes de prueba con ambos conjuntos de vectores sobre las
  mismas fotos neutras de referencia, para comparación visual.
- [ ] **[manual]** Evaluar si las emociones generadas con los nuevos vectores son
  visualmente más convincentes.

### 2.6 Reporte al director

- [ ] **[manual]** Preparar comparativas lado a lado: misma imagen de entrada, vectores
  originales vs. vectores con augmentation, para cada emoción.
- [ ] **[mail]** Enviar resultados y comparativas al director de tesis.

---

## Notas y decisiones pendientes

> Anotar acá cualquier pregunta o decisión que haya quedado abierta.

- [x] Definir cuántas imágenes de CelebA usar en el Bloque 1 → **arrancar con 20, escalar a 50 según tiempo de ejecución**.
- [ ] Definir el desvío estándar del ruido gaussiano a usar en augmentation (experimentar
  con σ=5, σ=10, σ=20 y elegir visualmente).
- [ ] Decidir si el flip horizontal tiene sentido para las emociones (la asimetría facial
  podría afectar el resultado — consultar con el director antes de incluirlo).
- [ ] Decidir si se van a combinar las tres augmentations o probarlas por separado para
  entender cuál aporta más.
- [ ] Decidir cómo hacer la máscara de cara para el reemplazo de fondo en CelebA (umbral
  simple vs. modelo de segmentación). Evaluar `rembg` como opción liviana.
