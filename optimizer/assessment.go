// Package optimizer provides prompt optimization capabilities for Language Learning Models.
package optimizer

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/yockii/gollm_cn/llm"
)

// assessPrompt evaluates a prompt's quality and effectiveness using the configured LLM.
// It performs a comprehensive analysis considering multiple factors including custom metrics,
// optimization goals, and historical context.
//
// The assessment process:
// 1. Constructs an evaluation prompt incorporating task description and history
// 2. Requests LLM evaluation of the prompt
// 3. Parses and validates the assessment response
// 4. Normalizes grading for consistency
//
// Parameters:
//   - ctx: Context for cancellation and timeout
//   - prompt: The prompt to be assessed
//
// Returns:
//   - OptimizationEntry containing the assessment results
//   - Error if assessment fails
//
// The assessment evaluates:
//   - Custom metrics specified in PromptOptimizer
//   - Prompt strengths with examples
//   - Weaknesses with improvement suggestions
//   - Overall effectiveness and efficiency
//   - Alignment with optimization goals
func (po *PromptOptimizer) assessPrompt(ctx context.Context, prompt *llm.Prompt) (OptimizationEntry, error) {
	recentHistory := po.recentHistory()
	assessPrompt := llm.NewPrompt(fmt.Sprintf(`
		评估以下针对任务的提示词: %s

		完整提示词结构:
		%+v

		最近历史记录:
		%+v

		自定义指标: %v

		优化目标: %s

		请在评估时考虑最近的历史记录。
		以 JSON 对象的形式提供你的评估，结构如下:
		{
			"metrics": [{"name": string, "value": number, "reasoning": string}, ...],
			"strengths": [{"point": string, "example": string}, ...],
			"weaknesses": [{"point": string, "example": string}, ...],
			"suggestions": [{"description": string, "expectedImpact": number, "reasoning": string}, ...],
			"overallScore": number,
			"overallGrade": string,
			"efficiencyScore": number,
			"alignmentWithGoal": number
		}

		重要提示: 
		- 请勿在你的回复中使用任何 Markdown 格式、代码块或反引号。
		- 仅返回原始 JSON 对象。
		- 对于数值评分，请使用 0 到 20（含）的等级。
		- 对于 overallGrade:
		  - 如果使用字母等级，请使用以下等级之一: F, D, C, B, A, A+
		  - 如果使用数字等级，请使用与 overallScore 相同的值 (0-20)
		- 每个数组（metrics、strengths、weaknesses、suggestions）中至少包含一个项目。
		- 为每个要点提供具体的例子和理由。
		- 评价提示词的效率以及与优化目标的一致性。
		- 根据建议的预期影响对建议进行排序（20 为最高影响）。
		- 在你的评估中使用清晰、无术语的语言。
		- 在提交之前，请仔细检查您的回复是否为有效的 JSON。
	`, po.taskDesc, prompt, recentHistory, po.customMetrics, po.optimizationGoal))

	// Generate assessment using LLM
	response, err := po.llm.Generate(ctx, assessPrompt)
	if err != nil {
		return OptimizationEntry{}, fmt.Errorf("failed to assess prompt: %w", err)
	}

	// Parse and validate assessment response
	cleanedResponse := cleanJSONResponse(response)
	var assessment PromptAssessment
	err = json.Unmarshal([]byte(cleanedResponse), &assessment)
	if err != nil {
		po.debugManager.LogResponse(fmt.Sprintf("Raw response: %s", response))
		return OptimizationEntry{}, fmt.Errorf("failed to parse assessment response: %w", err)
	}

	if err := llm.Validate(assessment); err != nil {
		return OptimizationEntry{}, fmt.Errorf("invalid assessment structure: %w", err)
	}

	// Normalize grading for consistency
	assessment.OverallGrade, err = normalizeGrade(assessment.OverallGrade, assessment.OverallScore)
	if err != nil {
		return OptimizationEntry{}, fmt.Errorf("invalid overall grade: %w", err)
	}

	return OptimizationEntry{
		Prompt:     prompt,
		Assessment: assessment,
	}, nil
}

// isOptimizationGoalMet determines if a prompt's assessment meets the configured
// optimization threshold. It supports both numerical and letter-based grading systems.
//
// For numerical ratings:
// - Uses a 0-20 scale
// - Compares against the configured threshold
//
// For letter grades:
// - Converts letter grades to GPA scale (0.0-4.3)
// - Requires A- (3.7) or better to meet goal
//
// Parameters:
//   - assessment: The PromptAssessment to evaluate
//
// Returns:
//   - bool: true if optimization goal is met
//   - error: if rating system is invalid or grade cannot be evaluated
//
// Example threshold values:
//   - Numerical: 0.75 requires score >= 15/20
//   - Letter: Requires A- or better
func (po *PromptOptimizer) isOptimizationGoalMet(assessment PromptAssessment) (bool, error) {
	if po.ratingSystem == "" {
		return false, nil
	}

	switch po.ratingSystem {
	case "numerical":
		return assessment.OverallScore >= 20*po.threshold, nil
	case "letter":
		gradeValues := map[string]float64{
			"A+": 4.3, "A": 4.0, "A-": 3.7,
			"B+": 3.3, "B": 3.0, "B-": 2.7,
			"C+": 2.3, "C": 2.0, "C-": 1.7,
			"D+": 1.3, "D": 1.0, "D-": 0.7,
			"F": 0.0,
		}
		gradeValue, exists := gradeValues[assessment.OverallGrade]
		if !exists {
			return false, fmt.Errorf("invalid grade: %s", assessment.OverallGrade)
		}
		return gradeValue >= 3.7, nil // Equivalent to A- or better
	default:
		return false, fmt.Errorf("unknown rating system: %s", po.ratingSystem)
	}
}
