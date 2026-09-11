# Guía de contribución

Reglas mínimas para mantener el repositorio ordenado y fácil de seguir entre todo el equipo.

## Ramas

- `main` siempre debe quedar en un estado funcional (el código corre sin errores).
- Trabaja en una rama nueva por tarea o feature, con el prefijo que corresponda:
  - `feat/nombre-corto` — nueva funcionalidad
  - `fix/nombre-corto` — corrección de un bug
  - `refactor/nombre-corto` — cambios internos sin alterar comportamiento
  - `docs/nombre-corto` — cambios solo de documentación
- Cuando termines, abre un Pull Request hacia `main` en vez de hacer push directo, para que alguien más del equipo pueda revisar el cambio.

## Commits

Este repo ya usa [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/); sigue el mismo formato:

```
<tipo>(<alcance opcional>): <descripción corta en presente>
```

Tipos usados en el proyecto:

| Tipo       | Cuándo usarlo                                              |
|------------|-------------------------------------------------------------|
| `feat`     | Agregar una funcionalidad nueva                              |
| `fix`      | Corregir un bug                                              |
| `refactor` | Reestructurar código sin cambiar su comportamiento           |
| `docs`     | Cambios en documentación (README, comentarios, etc.)         |
| `chore`    | Tareas de mantenimiento (gitignore, dependencias, config)    |
| `test`     | Agregar o modificar pruebas                                  |

Ejemplos:

```
feat(transform): agregar extracción de features de fecha
fix(loader): manejar archivo csv vacío
docs: actualizar instrucciones de instalación en README
```

Reglas rápidas:
- Un commit = un cambio lógico. Evita commits gigantes que mezclen varias cosas.
- Descripción en minúscula, sin punto final, en modo imperativo/presente ("agregar", no "agregado" ni "agregando").

## Pull Requests

- Título claro que resuma el cambio (puede seguir el mismo formato de los commits).
- Describe brevemente **qué** cambia y **por qué**.
- Antes de mergear, verifica que `python main.py` corre sin errores.

## Estilo de código

- Nombres de clases en `PascalCase`, funciones y variables en `snake_case`.
- Cada clase con una responsabilidad clara (ver `DataLoader`, `DataCleaner`, `DataTransform` como referencia).
- Evita dejar código comentado o archivos de prueba sueltos en el commit final; usa `.gitignore` para lo que no debe versionarse.
