# Risk Baseline / Parameters 更新申请

**重要提示**: 此 PR 模板专用于 risk baseline 或 gating parameters 的更新。任何对以下文件��修改必须使用此模板：
- `config/adaptive_gating_params.json`
- `baselines/risk/` 目录下的文件
- 其他风险相关的配置文件

---

## 📋 更新信息

**更新类型**:
- [ ] 参数再标定（来自 drift detection 触发）
- [ ] 手动调整（请说明原因）
- [ ] 紧急修复（请说明原因）

**更新文件**:
```
（列出修改的文件路径）
```

---

## 🔍 证据与报告

### Drift Detection Report（如果适用）

**漂移状态**: RED / YELLOW / GREEN
**触发条件**: （例如：连续 3 次 RED 检测）

**Drift Report 附件**: （链接或附件）
- [ ] 已上传 drift report JSON
- [ ] 已上传 drift detection 日志

### Regression Gates 结果

**SPL-4 Risk Regression**:
- [ ] Envelope Guard: ✓ PASS / ✗ FAIL
- [ ] Spike Risk Guard: ✓ PASS / ✗ FAIL

**SPL-5 CI Gate**:
- [ ] Portfolio Guards: ✓ PASS / ✗ FAIL
- [ ] Co-crash Guard: ✓ PASS / ✗ FAIL
- [ ] Correlation Guard: ✓ PASS / ✗ FAIL

**Regression Report 附件**: （链接或附件）
- [ ] 已上传 regression test 报告

### 三组对照结果（必填）

| 组别 | 配置 | 总收益 | CVaR-95 | Max DD | Co-crash |
|------|------|--------|---------|--------|----------|
| Group A | Baseline (No gating) | | | | |
| Group B | SPL-4 (Fixed gating) | | | | |
| Group C | Candidate (新参数) | | | | |

**Trade-off 分析**:
- 收益变化: (+/-) ____ %
- 风险变化: (+/-) ____ %
- 协同变化: (+/-) ____ 次

**Comparison Report 附件**: （链接或附件）
- [ ] 已上传三组对照报告

---

## 🎯 更新理由

**为什么需要更新**:
```
（请详细说明：
1. 当前 baseline 存在的问题
2. 新参数如何解决这些问题
3. 为什么其他方案不可行
）
```

**预期效果**:
```
（请说明：
1. 预期的收益/风险改善
2. 是否有副作用
3. 如何监控副作用
）
```

---

## ⚠️ 风险评估

**潜在风险**:
- [ ] 无显著风险
- [ ] 有潜在风险（请说明）

**缓解措施**:
```
（如果有风险，说明如何缓解）
```

**回滚计划**:
```
（如果更新失败，如何回滚到旧版本）
- 旧参数 commit hash: _______
- 回滚步骤: _____________
）
```

---

## 👥 审批与确认

**自检清单**:
- [ ] 我已确认所有 regression gates 通过
- [ ] 我已确认新参数不会导致 envelope/spike/co-crash 退化
- [ ] 我已生成并审阅了三组对照报告
- [ ] 我已准备好回滚计划
- [ ] 我理解此更新将在合并后立即生效

**需要审批**:
- [ ] Risk Team Lead: _______ (Approval / Reject)
- [ ] Tech Lead: _______ (Approval / Reject)

**审批意见**:
```
（审批人填写）

审批结果: [ ] APPROVED / [ ] REJECTED / [ ] REQUEST CHANGES

审批理由:
（请说明）

Additional Comments:
（如有）
```

---

## 📎 附件清单

请确保上传以下附件（可在 PR comment 中附链接）：

1. [ ] Drift Report JSON (`drift_report.json`)
2. [ ] Regression Test Report (`regression_report.json`)
3. [ ] 三组对照报告 (`comparison_report.md`)
4. [ ] 候选参数文件 (`candidate_params.json`)
5. [ ] 审查清单 (`review_checklist.md`)

---

## 🔄 审批流程

1. **作者提交**: 作者填写此模板并提交 PR
2. **自动检查**: CI 会自动运行 regression gates，任何 FAIL 将阻断 PR
3. **人工审阅**: Risk Team Lead 审查证据和报告
4. **合并决定**: 审批通过后，由 maintainer 合并
5. **合并后监控**: 至少监控 7 天，如出现问题立即回滚

---

**提交前请确认**:
- 所有必须的附件已上传
- Regression gates 全部 PASS
- 至少一位审批人已 APPROVE
- 回滚计划已准备好

---

*此模板确保 baseline 更新有完整的审计追踪和证据链。*
