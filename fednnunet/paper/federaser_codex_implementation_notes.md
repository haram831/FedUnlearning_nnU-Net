# FedEraser 핵심 구현 참고 노트 for Codex CLI

> 목적: 이 문서는 Codex CLI가 FedEraser 논문의 핵심 구현부를 코드로 옮길 때 참고하도록 만든 구현 명세이다.  
> 대상 논문: *Federated Unlearning* / FedEraser.  
> 구현 범위: **client-level federated unlearning**, 즉 특정 client `ku`의 전체 데이터 영향력을 기존 FL global model에서 제거하는 절차.

---

## 1. FedEraser가 해결하려는 문제

표준 Federated Learning에서는 여러 client가 각 round마다 local training을 수행하고, server가 client update를 aggregation하여 global model을 갱신한다.

문제는 특정 client `ku`가 삭제 요청을 했을 때, 이미 학습된 global model `M` 안에 `ku`의 데이터 영향이 남아 있다는 점이다. 가장 정확한 방법은 target client를 제외하고 처음부터 다시 학습하는 `FedRetrain`이지만, 이는 시간이 너무 오래 걸린다.

FedEraser의 목표는 다음과 같다.

```text
기존 global model에서 target client ku의 영향을 제거하되,
처음부터 재학습하는 것보다 훨씬 빠르게 unlearned model Mf를 구성한다.
```

핵심 아이디어는 다음 두 가지이다.

1. **FL 학습 중 server가 client update를 일정 간격으로 저장한다.**
2. 삭제 요청이 오면 target client를 제외한 나머지 client들의 저장된 update를 calibration하여 unlearned model을 재구성한다.

---

## 2. 중요한 변수와 기호

| 기호 / 변수 | 코드 변수 예시 | 의미 |
|---|---|---|
| `K` | `num_clients` | 전체 federated client 수 |
| `C_k` | `client[k]` | k번째 client |
| `ku` | `target_client_id` | 삭제 대상 client index |
| `kc` | `calibrating_client_id` | calibration에 참여하는 client, 즉 `kc != ku` |
| `E` | `num_global_rounds` | 원래 FL 학습의 전체 global round 수 |
| `Elocal` | `local_epochs` | 표준 FL에서 client가 수행하는 local epoch 수 |
| `Ecali` | `calibration_epochs` | FedEraser calibration training에서 client가 수행하는 local epoch 수 |
| `r` | `calibration_ratio` | `r = Ecali / Elocal` |
| `Δt` | `retain_interval` | server가 client update를 저장하는 round 간격 |
| `T` | `num_retained_rounds` | 저장된 retained round 수, 대략 `floor(E / Δt)` |
| `M_i` | `global_model` | i번째 round의 global model |
| `Mf_tj` | `unlearned_model` | FedEraser가 재구성 중인 calibrated/unlearned global model |
| `U_k^tj` | `retained_update[k][tj]` | 원래 FL 학습 중 round `tj`에서 저장한 client k의 update |
| `Û_kc^tj` | `current_update` | calibration training 후 새로 계산한 calibrating client의 update |
| `Ũ_kc^tj` | `calibrated_update` | FedEraser가 보정한 client update |
| `w_kc` | `client_weight[kc]` | aggregation에서 사용하는 client weight, 보통 local sample 수 기반 |

---

## 3. 원래 FL 학습 중 추가해야 하는 저장 기능

FedEraser는 unlearning 시점에 historical client update가 필요하다. 따라서 표준 FL 학습 과정 중 server는 일정 간격 `Δt`마다 각 client의 update를 저장해야 한다.

### 3.1 저장 대상

각 retained round `tj`에 대해 다음을 저장한다.

```text
round_id = tj
client_id = k
update = U_k^tj
client_weight = w_k
optional: num_samples = N_k
```

### 3.2 retained round 정의

논문에서는 다음과 같이 둔다.

```text
t1 = 1
t_{j+1} = t_j + Δt
T = floor(E / Δt)
```

구현에서는 zero-based round를 사용할 수 있으므로 주의한다.

예시:

```python
if round_idx % retain_interval == 0:
    save_client_updates(round_idx, client_updates)
```

단, 논문 표기와 코드의 round numbering이 다르면 `round_idx + 1` 기준으로 맞출지 명확히 정해야 한다.

### 3.3 update 정의

client update는 보통 다음 중 하나로 정의할 수 있다.

```text
update = local_model_params_after_training - global_model_params_before_training
```

FedEraser의 model update 식은 다음과 같다.

```text
M_{next} = M_current + aggregated_update
```

따라서 기존 코드가 gradient 또는 delta를 반대 부호로 저장하고 있다면 반드시 부호를 확인해야 한다.

---

## 4. FedEraser 전체 알고리즘 흐름

FedEraser는 삭제 요청이 들어온 뒤 다음 절차로 unlearned model `Mf`를 만든다.

```text
Input:
- initial global model M1
- retained client updates U
- target client index ku
- number of retained/global calibration rounds T
- calibration epochs Ecali

Output:
- unlearned global model Mf
```

전체 흐름:

```text
1. Mf를 초기 global model M1로 초기화한다.
2. 저장된 retained round를 순서대로 순회한다.
3. 각 retained round tj마다 target client ku를 제외한 client들을 calibration client로 사용한다.
4. 각 calibrating client kc는 현재 Mf를 받아 Ecali epoch만큼 local training한다.
5. client는 current update Û_kc^tj를 계산하여 server로 보낸다.
6. server는 저장된 historical update U_kc^tj의 norm과 current update Û_kc^tj의 direction을 결합하여 calibrated update Ũ_kc^tj를 만든다.
7. server는 calibrated updates를 weighted average로 aggregation한다.
8. Mf를 aggregated calibrated update로 갱신한다.
9. 모든 retained round를 처리하면 최종 Mf를 반환한다
```

---

## 5. 핵심 구현부 1: Calibration Training

### 5.1 역할

Calibration training은 target client를 제외한 client들이 현재 unlearned model `Mf_tj`를 기준으로 짧은 local training을 수행하여 새로운 update direction을 얻는 과정이다.

표준 FL client training 함수를 거의 재사용하면 된다.

### 5.2 입력

```python
client: Client
model: torch.nn.Module        # current unlearned model Mf_tj
calibration_epochs: int       # Ecali
train_loader: DataLoader      # client kc의 local data
optimizer_config: dict
```

### 5.3 출력

```python
current_update: StateDictLike
```

여기서 `current_update`는 다음과 같이 계산한다.

```python
current_update = trained_local_model_params - initial_model_params
```

### 5.4 구현 시 주의

- target client `ku`는 calibration training에 절대 참여하면 안 된다.
- calibration training은 원래 `Elocal`보다 적은 epoch인 `Ecali`만 수행한다.
- `Ecali = int(calibration_ratio * local_epochs)`로 계산할 경우 최소 1 epoch 보장이 필요한지 결정한다.
- optimizer, learning rate, batch size는 기존 FL local training 설정과 가능한 동일하게 둔다.

---

## 6. 핵심 구현부 2: Update Calibration

FedEraser의 가장 중요한 부분이다.

논문 식은 다음과 같다.

```text
Ũ_kc^tj = ||U_kc^tj|| * (Û_kc^tj / ||Û_kc^tj||)
```

의미:

```text
calibrated update = historical update의 크기(norm) + current update의 방향(direction)
```

즉, 원래 학습 중 저장한 update `U_kc^tj`는 target client의 영향을 받은 global model에서 나온 것이므로 그대로 쓰면 안 된다. 대신 calibration training으로 얻은 새 update `Û_kc^tj`의 방향을 사용한다. 하지만 크기는 원래 update의 norm을 유지한다.

### 6.1 코드 수준 구현

PyTorch state_dict 기준으로 구현하면 다음 유틸리티가 필요하다.

```python
def state_dict_l2_norm(update: dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute global L2 norm over all tensor values in an update."""


def normalize_state_dict(update: dict[str, torch.Tensor], eps: float = 1e-12) -> dict[str, torch.Tensor]:
    """Return update / ||update||."""


def scale_state_dict(update: dict[str, torch.Tensor], scale: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return scale * update."""


def calibrate_update(
    retained_update: dict[str, torch.Tensor],
    current_update: dict[str, torch.Tensor],
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    retained_norm = state_dict_l2_norm(retained_update)
    current_norm = state_dict_l2_norm(current_update)
    return {
        name: retained_norm * tensor / (current_norm + eps)
        for name, tensor in current_update.items()
    }
```

### 6.2 layer별 norm vs 전체 norm

논문 식은 update 전체에 대한 norm처럼 표현되어 있다. 구현은 두 가지 방식이 가능하다.

권장 기본값:

```text
모든 parameter tensor를 flatten한 global L2 norm을 사용한다.
```

대안:

```text
각 layer별 norm을 따로 적용한다.
```

Codex는 우선 global L2 norm 방식으로 구현하고, layer-wise calibration은 옵션으로 분리한다.

### 6.3 zero norm 처리

`current_update`의 norm이 0에 가까우면 division 문제가 생긴다.

권장 처리:

```python
if current_norm < eps:
    return zeros_like(current_update)
```

또는 retained update를 그대로 쓰지 말고 해당 client update를 skip한다. 기본 구현은 안정성을 위해 zero update 반환을 권장한다.

---

## 7. 핵심 구현부 3: Calibrated Update Aggregation

논문 식은 다음과 같다.

```text
Ũ^tj = 1 / ((K - 1) * Σ w_kc) * Σ w_kc * Ũ_kc^tj
```

다만 실제 FL 구현에서는 보통 다음 방식이 더 일반적이다.

```text
aggregated_update = Σ (N_kc / Σ N_kc) * calibrated_update_kc
```

논문의 `w_kc` 정의가 local sample count 기반 weight로 설명되므로, 구현에서는 다음을 권장한다.

```python
def aggregate_updates(
    calibrated_updates: dict[int, StateDict],
    client_num_samples: dict[int, int],
) -> StateDict:
    total = sum(client_num_samples[k] for k in calibrated_updates.keys())
    aggregated = zeros_like(first_update)
    for client_id, update in calibrated_updates.items():
        alpha = client_num_samples[client_id] / total
        aggregated += alpha * update
    return aggregated
```

### 주의

- target client `ku`의 update는 aggregation에 포함하지 않는다.
- retained update 저장 시 client별 sample 수 또는 aggregation weight도 같이 저장하는 것이 좋다.
- 기존 FL 코드가 FedAvg aggregation 함수를 가지고 있다면 최대한 재사용한다.

---

## 8. 핵심 구현부 4: Unlearned Model Updating

논문 식은 다음과 같다.

```text
Mf_{tj+1} = Mf_tj + Ũ^tj
```

구현 함수 예시:

```python
def apply_update_to_model(model: torch.nn.Module, update: StateDict) -> None:
    with torch.no_grad():
        params = model.state_dict()
        for name in params:
            if name in update:
                params[name].add_(update[name].to(params[name].device))
        model.load_state_dict(params)
```

주의:

- BatchNorm의 running statistics 같은 buffer를 update에 포함할지 결정해야 한다.
- 일반적으로 trainable parameter만 update 대상으로 삼는 것이 안전하다.
- `state_dict()` 전체를 다루는 경우 dtype/device mismatch를 조심한다.

---

## 9. 구현용 메인 함수 설계

권장 파일:

```text
src/unlearning/federaser.py
```

권장 함수:

```python
def federaser_unlearn(
    initial_model: torch.nn.Module,
    retained_updates: list[RetainedRound],
    clients: list[Client],
    target_client_id: int,
    calibration_epochs: int,
    device: torch.device,
    aggregation: str = "fedavg",
) -> torch.nn.Module:
    """Run FedEraser client-level unlearning."""
```

권장 dataclass:

```python
@dataclass
class ClientUpdateRecord:
    client_id: int
    update: dict[str, torch.Tensor]
    num_samples: int


@dataclass
class RetainedRound:
    round_id: int
    client_updates: dict[int, ClientUpdateRecord]
```

메인 pseudocode:

```python
def federaser_unlearn(...):
    unlearned_model = deepcopy(initial_model)

    for retained_round in retained_updates:
        calibrated_updates = {}
        client_num_samples = {}

        for client in clients:
            if client.client_id == target_client_id:
                continue

            retained_record = retained_round.client_updates[client.client_id]

            current_update = run_calibration_training(
                client=client,
                model=deepcopy(unlearned_model),
                epochs=calibration_epochs,
                device=device,
            )

            calibrated_update = calibrate_update(
                retained_update=retained_record.update,
                current_update=current_update,
            )

            calibrated_updates[client.client_id] = calibrated_update
            client_num_samples[client.client_id] = retained_record.num_samples

        aggregated_update = aggregate_updates(
            calibrated_updates,
            client_num_samples,
        )

        apply_update_to_model(unlearned_model, aggregated_update)

    return unlearned_model
```

---

## 10. 기존 FL 학습 코드에 추가할 hook

FedEraser를 구현하려면 표준 FL training loop에 update 저장 hook이 필요하다.

권장 파일:

```text
src/federated/trainer.py
src/federated/checkpoint.py
```

예시:

```python
for round_idx in range(num_global_rounds):
    global_params_before = copy_model_params(global_model)
    client_updates = {}

    for client in selected_clients:
        local_model = deepcopy(global_model)
        train_local(local_model, client.train_loader, local_epochs)
        update = subtract_state_dict(local_model.state_dict(), global_params_before)
        client_updates[client.id] = update

    aggregated_update = fedavg(client_updates, client_weights)
    apply_update_to_model(global_model, aggregated_update)

    if should_retain_round(round_idx, retain_interval):
        save_retained_round(
            round_id=round_idx,
            client_updates=client_updates,
            client_num_samples=client_num_samples,
            path=retained_update_dir,
        )
```

---

## 11. CLI 인자 설계

사용자가 이미 `Δt`와 `r`을 CLI에서 받고 있다면 다음 인자를 명확히 연결한다.

```text
--retain-interval Δt
--calibration-ratio r
--local-epochs Elocal
--target-client-id ku
--retained-update-dir PATH
--initial-model-checkpoint PATH
--output-unlearned-checkpoint PATH
```

`Ecali` 계산:

```python
calibration_epochs = max(1, int(round(calibration_ratio * local_epochs)))
```

논문 실험 기본값:

```text
retain_interval Δt = 2
calibration_ratio r = 0.5
```

이 경우 기대 속도 향상은 대략 다음과 같다.

```text
speedup ≈ Δt * (1 / r)
```

즉 `Δt = 2`, `r = 0.5`이면 이론적으로 약 `4x` speed-up을 기대한다.

---

## 12. 평가 지표 구현

FedEraser 구현 후 다음 지표를 저장한다.

### 12.1 기본 성능

```text
- test accuracy
- test loss
- target client data accuracy
- target client data loss
- unlearning time
```

### 12.2 prediction difference

논문에서 제시한 target data 기준 prediction difference:

```text
P_diss = (1 / N) * Σ || M(x_i) - Mf(x_i) ||_2,  x_i ∈ D_ku
```

구현 예시:

```python
def prediction_difference(original_model, unlearned_model, target_loader):
    diffs = []
    for x, _ in target_loader:
        p_orig = softmax(original_model(x), dim=-1)
        p_unlearn = softmax(unlearned_model(x), dim=-1)
        diffs.append(torch.norm(p_orig - p_unlearn, p=2, dim=-1))
    return torch.cat(diffs).mean().item()
```

### 12.3 retrained model과의 비교

가능하다면 `FedRetrain` baseline을 만들어 다음을 비교한다.

```text
- FedEraser vs FedRetrain test accuracy 차이
- FedEraser vs FedRetrain target accuracy 차이
- FedEraser vs FedRetrain parameter deviation
- FedEraser unlearning time vs FedRetrain retraining time
```

---

## 13. 테스트 전략

Codex는 구현 후 최소한 다음 테스트를 추가한다.

### 13.1 update utility test

파일:

```text
tests/test_state_dict_ops.py
```

검증:

```text
- subtract_state_dict가 올바른 delta를 만든다.
- state_dict_l2_norm이 tensor들을 flatten한 전체 norm과 일치한다.
- apply_update_to_model이 parameter를 정확히 더한다.
```

### 13.2 calibration test

파일:

```text
tests/test_federaser_calibration.py
```

검증:

```text
- calibrated_update의 norm이 retained_update의 norm과 거의 같다.
- calibrated_update의 direction이 current_update의 direction과 같다.
- current_update norm이 0이면 NaN이 생기지 않는다.
```

### 13.3 target client exclusion test

검증:

```text
- target_client_id가 calibration training에 사용되지 않는다.
- target_client_id의 retained update가 aggregation에 포함되지 않는다.
```

### 13.4 smoke test

작은 toy model과 synthetic clients를 사용하여 다음을 확인한다.

```text
- federaser_unlearn 함수가 에러 없이 실행된다.
- output model checkpoint가 생성된다.
- unlearning report JSON이 생성된다.
```

---

## 14. 저장 파일 / 산출물 권장 구조

```text
checkpoints/
  global_round_0001.pth
  global_round_0002.pth

retained_updates/
  round_0001/
    client_000_update.pth
    client_001_update.pth
    metadata.json
  round_0003/
    client_000_update.pth
    client_001_update.pth
    metadata.json

unlearning_runs/
  forget_client_003/
    unlearned_model.pth
    unlearning_report.json
    metrics.json
```

`metadata.json` 예시:

```json
{
  "round_id": 1,
  "retain_interval": 2,
  "clients": {
    "0": { "num_samples": 120, "update_path": "client_000_update.pth" },
    "1": { "num_samples": 118, "update_path": "client_001_update.pth" }
  }
}
```

`unlearning_report.json` 예시:

```json
{
  "method": "FedEraser",
  "target_client_id": 3,
  "retain_interval": 2,
  "calibration_ratio": 0.5,
  "local_epochs": 10,
  "calibration_epochs": 5,
  "num_retained_rounds": 50,
  "unlearning_time_sec": 123.4,
  "output_checkpoint": "unlearned_model.pth"
}
```

---

## 15. Codex에게 줄 구현 프롬프트 예시

```text
Read this file first: docs/federaser_codex_implementation_notes.md.

Implement FedEraser client-level unlearning based on the notes.
Do not implement unrelated algorithms.

First, inspect the existing FL training code and identify:
1. where client updates are computed,
2. where FedAvg aggregation is performed,
3. where checkpoints are saved,
4. how client sample counts are represented.

Before editing files, report the exact files and functions you plan to modify.
```

단계별 구현 프롬프트:

```text
Implement only the retained update saving hook for FedEraser.
Use --retain-interval to decide which rounds to save.
Save each retained client update and metadata containing round_id, client_id, num_samples, and update path.
Add tests if possible.
```

```text
Now implement the core FedEraser utilities:
- state_dict_l2_norm
- subtract_state_dict
- calibrate_update
- aggregate_updates
- apply_update_to_model
Add unit tests for each function.
```

```text
Now implement federaser_unlearn().
It must exclude target_client_id from calibration and aggregation.
It must use calibration_epochs = max(1, round(calibration_ratio * local_epochs)).
It must save unlearned_model.pth and unlearning_report.json.
```

---

## 16. 구현 시 헷갈리기 쉬운 부분

### 16.1 FedEraser는 target client의 local data를 사용하지 않는다

Unlearning 과정에서 target client `ku`는 제외된다. target client의 update나 data를 calibration에 사용하면 안 된다.

### 16.2 retained update는 그대로 누적하면 FedAccum이다

저장된 update를 calibration 없이 단순히 누적하면 논문의 비교 방법인 `FedAccum`에 가깝다. FedEraser의 핵심은 반드시 다음 calibration을 수행하는 것이다.

```text
historical norm + current direction
```

### 16.3 initial model이 필요하다

FedEraser는 최종 global model에서 바로 시작하는 것이 아니라, 원래 FL의 initial global model `M1`에서 시작하여 retained updates를 순차적으로 적용해 unlearned model을 재구성한다.

### 16.4 첫 reconstruction epoch 처리

논문은 첫 reconstruction epoch에서는 initial model이 아직 target client의 영향을 받지 않았기 때문에 별도 calibration 없이 update할 수 있다고 설명한다. 구현에서는 단순성과 일관성을 위해 모든 retained round에 calibration을 적용해도 되지만, 논문에 더 가깝게 하려면 첫 retained round에 대해 calibration skip 옵션을 둘 수 있다.

권장 옵션:

```python
skip_first_calibration: bool = False
```

기본은 `False`로 두고, 논문 재현 실험에서는 `True`를 실험 옵션으로 둔다.

### 16.5 Δt가 클수록 빠르지만 부정확할 수 있다

`retain_interval`이 커지면 저장하는 round 수가 줄어 unlearning은 빨라진다. 하지만 target client의 영향이 더 많이 남을 수 있으므로 accuracy와 forgetting 효과를 같이 봐야 한다.

### 16.6 r이 작을수록 빠르지만 direction 추정이 부정확할 수 있다

`calibration_ratio`가 작으면 calibration training 시간이 줄어든다. 그러나 current update direction의 품질이 낮아질 수 있다.

---

## 17. 최소 구현 체크리스트

- [ ] 표준 FL 학습 중 retained update 저장 기능 추가
- [ ] `retain_interval` CLI 인자 추가 또는 기존 인자 연결
- [ ] `calibration_ratio` CLI 인자 추가 또는 기존 인자 연결
- [ ] `target_client_id` CLI 인자 추가
- [ ] `Ecali = max(1, round(r * Elocal))` 계산
- [ ] target client 제외 로직 구현
- [ ] calibration training 구현 또는 기존 local training 재사용
- [ ] retained update norm 계산
- [ ] current update direction 계산
- [ ] calibrated update 생성
- [ ] calibrated FedAvg aggregation 구현
- [ ] unlearned model sequential update 구현
- [ ] unlearned checkpoint 저장
- [ ] unlearning report 저장
- [ ] unit test 추가
- [ ] small toy FL smoke test 추가

---

## 18. 논문 기반 구현 요약

FedEraser는 다음 한 줄로 요약할 수 있다.

```text
Store historical client updates during FL, and when a client must be forgotten,
reconstruct the global model from the initial model by applying calibrated updates
from only the remaining clients.
```

가장 중요한 구현 식은 다음이다.

```text
calibrated_update = norm(retained_update) * normalize(current_calibration_update)
```

그리고 최종 모델은 다음처럼 반복 갱신된다.

```text
unlearned_model = unlearned_model + aggregate(calibrated_updates_from_remaining_clients)
```

