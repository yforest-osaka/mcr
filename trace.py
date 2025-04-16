import collections
import sys

call_counts = collections.Counter()


def trace_calls(frame, event, arg):
    if event != "call":
        return

    code = frame.f_code
    func_name = code.co_name
    filename = code.co_filename

    # 除外条件
    if (
        func_name.startswith("_")
        or func_name.startswith("<")
        or filename.startswith("<")  # ← これを追加
        or "site-packages" in filename
        or "lib/python" in filename
    ):
        return

    call_counts[(func_name, filename)] += 1
    return trace_calls


def analyze_main():
    import main  # 実行対象の main.py

    main.main()


if __name__ == "__main__":
    sys.settrace(trace_calls)
    analyze_main()
    sys.settrace(None)

    print("\n=== 呼び出された関数一覧（回数順） ===")
    for (func_name, filename), count in call_counts.most_common():
        print(f"{func_name:<30} ({filename}) - {count} 回")
