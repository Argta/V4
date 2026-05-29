classdef binaural_gui < matlab.apps.AppBase
    % BINAURAL_GUI  双耳声源定位仿真系统 v3.0
    %   交互式 3D 场景可视化与仿真控制

    properties (Access = private)
        proj_root   = ''
        results_dir = ''
        scene_list  = struct('name', {}, 'path', {})
        data        = struct()
        anim_idx    = 1
        anim_speed  = 1
        is_playing  = false

        h_source
        h_head_dir
        h_cursor3d
        h_overlay
        h_wave_cur
        h_anal_cur

        head_model = struct('vertices', [], 'faces', [], 'scale', 1)
        head_loaded = false

        itd_cache  = []
        ild_cache  = []
        t_analysis = []

        % v3.0: 定位评估数据
        doa_estimated = []
        doa_truth = []
        loc_timestamps = []
        eval_metrics = struct()
        doa_itd_only = []
        doa_ild_only = []
    end

    properties (Access = private, Transient, NonCopyable)
        UIFigure    matlab.ui.Figure
        SceneDrop   matlab.ui.control.DropDown
        RunBtn      matlab.ui.control.Button
        PlayBtn     matlab.ui.control.Button
        ResetBtn    matlab.ui.control.Button
        SpeedSlider matlab.ui.control.Slider
        ProgressSlider matlab.ui.control.Slider
        SpeedVal    matlab.ui.control.Label
        InfoText    matlab.ui.control.TextArea
        Axes3D      matlab.ui.control.UIAxes
        AxesWave    matlab.ui.control.UIAxes
        AxesWaveRaw matlab.ui.control.UIAxes
        WaveChannel matlab.ui.control.DropDown
        HrtfDrop    matlab.ui.control.DropDown
        SourceDrop  matlab.ui.control.DropDown
        ItdZoom     matlab.ui.control.Slider
        IldZoom     matlab.ui.control.Slider
        AxesAnal    matlab.ui.control.UIAxes
        AxesEvalDOA matlab.ui.control.UIAxes
        AxesEvalErr matlab.ui.control.UIAxes
        AxesEvalCM  matlab.ui.control.UIAxes
        EvalText    matlab.ui.control.TextArea
        StatusBar   matlab.ui.control.Label
        ProgressBar matlab.ui.control.Label
    end

    methods (Access = public)
        function app = binaural_gui()
            create_components(app);
            startupFcn(app);
        end
        function delete(app)
            app.is_playing = false;
            ts = timerfindall;
            for i = 1:length(ts)
                try stop(ts(i)); delete(ts(i)); catch; end
            end
        end
    end

    methods (Access = private)
        function create_components(app)
            screen = get(groot, 'ScreenSize');
            w = 1400; h = 850;
            left = max(0, (screen(3)-w)/2);
            bottom = max(0, (screen(4)-h)/2);

            app.UIFigure = uifigure('Name', '双耳声源定位仿真系统 v3.0', ...
                'Position', [left bottom w h], 'Color', [0.12 0.13 0.16], ...
                'CloseRequestFcn', @(fig,~) delete(fig));

            grid = uigridlayout(app.UIFigure, [3 2]);
            grid.RowHeight = {'1x', 22};
            grid.ColumnWidth = {340, '1x'};
            grid.Padding = [8 8 8 8];
            grid.RowSpacing = 6;
            grid.ColumnSpacing = 10;

            % ===== 左侧控制面板 =====
            left_panel = uipanel(grid, 'Title', ' 控制面板 ', ...
                'FontSize', 12, 'FontWeight', 'bold', ...
                'BackgroundColor', [0.13 0.14 0.18], ...
                'ForegroundColor', [0.9 0.9 0.9], ...
                'BorderType', 'none', 'HighlightColor', [0.2 0.22 0.28]);
            left_panel.Layout.Row = 1;
            left_panel.Layout.Column = 1;

            lg = uigridlayout(left_panel, [4 1]);
            lg.RowHeight = {135, 147, 149, '1x'};
            lg.Padding = [8 8 8 8];
            lg.RowSpacing = 6;

            % -- 场景选择 --
            sp = uipanel(lg, 'Title', ' 场景 ', ...
                'FontSize', 10, 'FontWeight', 'bold', ...
                'ForegroundColor', [1 1 1], ...
                'BackgroundColor', [0.15 0.16 0.20], ...
                'BorderType', 'none', 'HighlightColor', [1.0 0.55 0.1]);
            sg = uigridlayout(sp, [4 3]);
            sg.RowHeight = {'1x', 26, 28, '1x'};
            sg.ColumnWidth = {'1x', 50, 60};
            sg.Padding = [8 6 8 6];
            sg.RowSpacing = 4;
            sg.ColumnSpacing = 5;

            app.SceneDrop = uidropdown(sg, ...
                'Items', {'(请选择场景)'}, ...
                'Value', '(请选择场景)', 'FontSize', 10, ...
                'FontColor', [0.9 0.9 0.9], 'BackgroundColor', [0.2 0.22 0.28], ...
                'ValueChangedFcn', @(~,~) on_scene_select(app));
            app.SceneDrop.Layout.Row = 2;
            app.SceneDrop.Layout.Column = [1 3];

            rb = uibutton(sg, 'push', 'Text', char(8635), 'FontSize', 14, ...
                'Tooltip', '刷新场景列表', ...
                'ButtonPushedFcn', @(~,~) refresh_scenes(app));
            rb.Layout.Row = 3; rb.Layout.Column = 1;

            lb = uibutton(sg, 'push', 'Text', '加载数据', 'FontSize', 10, ...
                'Tooltip', '加载 .mat 结果文件', ...
                'ButtonPushedFcn', @(~,~) on_load(app));
            lb.Layout.Row = 3; lb.Layout.Column = [2 3];

            % -- 仿真控制 --
            sp2 = uipanel(lg, 'Title', ' 仿真 ', ...
                'FontSize', 10, 'FontWeight', 'bold', ...
                'ForegroundColor', [1 1 1], ...
                'BackgroundColor', [0.15 0.16 0.20], ...
                'BorderType', 'none', 'HighlightColor', [0.2 0.55 0.85]);
            sg2 = uigridlayout(sp2, [8 1]);
            sg2.RowHeight = {'1x', 30, 20, 20, 30, 30, 22, '1x'};
            sg2.Padding = [8 6 8 6];
            sg2.RowSpacing = 3;

            % HRTF 方案选择
            uilabel(sg2, 'Text', 'HRTF 方案:', 'FontSize', 9, ...
                'FontWeight', 'bold', 'HorizontalAlignment', 'left', ...
                'FontColor', [0.85 0.85 0.85]);
            app.HrtfDrop = uidropdown(sg2, ...
                'Items', {'analytical (球头模型)', ...
                          'parametric (+耳廓/耳道/肩膀)', ...
                          'measured (实测HRTF数据集)'}, ...
                'Value', 'analytical (球头模型)', 'FontSize', 9, ...
                'FontColor', [0.9 0.9 0.9], 'BackgroundColor', [0.2 0.22 0.28]);
            app.HrtfDrop.Layout.Row = 2;

            % 声源选择
            uilabel(sg2, 'Text', '声源类型:', 'FontSize', 9, ...
                'FontWeight', 'bold', 'HorizontalAlignment', 'left', ...
                'FontColor', [0.85 0.85 0.85]);
            app.SourceDrop = uidropdown(sg2, ...
                'Items', {'(使用YAML默认)', 'human_voice', 'sine', ...
                          'bowl_impact', 'chair_sliding'}, ...
                'Value', '(使用YAML默认)', 'FontSize', 9, ...
                'FontColor', [0.9 0.9 0.9], 'BackgroundColor', [0.2 0.22 0.28]);
            app.SourceDrop.Layout.Row = 3;

            app.RunBtn = uibutton(sg2, 'push', ...
                'Text', '▶  运行仿真', 'FontSize', 13, ...
                'FontWeight', 'bold', 'BackgroundColor', [0.2 0.55 0.85], ...
                'FontColor', 'white', ...
                'ButtonPushedFcn', @(~,~) on_run(app));
            app.RunBtn.Layout.Row = 5;

            ul = uilabel(sg2, 'Text', '', 'FontSize', 9, ...
                'HorizontalAlignment', 'center', 'FontColor', [0.85 0.85 0.85]);
            ul.Layout.Row = 6;

            % -- 动画控制 --
            sp3 = uipanel(lg, 'Title', ' 动画 ', ...
                'FontSize', 10, 'FontWeight', 'bold', ...
                'ForegroundColor', [1 1 1], ...
                'BackgroundColor', [0.15 0.16 0.20], ...
                'BorderType', 'none', 'HighlightColor', [0.15 0.70 0.25]);
            sg3 = uigridlayout(sp3, [5 5]);
            sg3.RowHeight = {'1x', 30, 26, 22, '1x'};
            sg3.ColumnWidth = {52, 52, 40, '1x', 35};
            sg3.Padding = [8 4 8 6];
            sg3.ColumnSpacing = 4;

            app.PlayBtn = uibutton(sg3, 'push', ...
                'Text', '播放', 'FontSize', 11, 'FontWeight', 'bold', ...
                'BackgroundColor', [0.15 0.70 0.25], 'FontColor', 'white', ...
                'ButtonPushedFcn', @(~,~) on_play_pause(app));
            app.PlayBtn.Layout.Row = 2;
            app.PlayBtn.Layout.Column = [1 2];

            app.ResetBtn = uibutton(sg3, 'push', ...
                'Text', '重置', 'FontSize', 9, ...
                'Tooltip', '重置动画', ...
                'ButtonPushedFcn', @(~,~) on_reset(app));
            app.ResetBtn.Layout.Row = 2;
            app.ResetBtn.Layout.Column = 3;

            ul_spd = uilabel(sg3, 'Text', '速度:', 'FontSize', 9, ...
                'HorizontalAlignment', 'right', ...
                'FontColor', [0.85 0.85 0.85]);
            ul_spd.Layout.Row = 2;
            ul_spd.Layout.Column = 4;

            app.SpeedSlider = uislider(sg3, ...
                'Limits', [0.2 5], 'Value', 1, 'MajorTicks', [0.2 1 2 3 4 5], ...
                'ValueChangedFcn', @(~,~) on_speed(app));
            app.SpeedSlider.Layout.Row = 3;
            app.SpeedSlider.Layout.Column = [2 4];

            app.SpeedVal = uilabel(sg3, 'Text', '1x', 'FontSize', 9, ...
                'FontWeight', 'bold', 'FontColor', [0.9 0.9 0.9]);
            app.SpeedVal.Layout.Row = 3;
            app.SpeedVal.Layout.Column = 5;

            app.ProgressSlider = uislider(sg3, ...
                'Limits', [0 1], 'Value', 0, 'Step', 0.001, ...
                'MajorTicks', [0 0.25 0.5 0.75 1], ...
                'ValueChangedFcn', @(~,~) on_progress(app), ...
                'ValueChangingFcn', @(~,~) on_progress(app));
            app.ProgressSlider.Layout.Row = 4;
            app.ProgressSlider.Layout.Column = [1 5];

            % -- 场景信息 --
            sp4 = uipanel(lg, 'Title', ' 场景信息 ', ...
                'FontSize', 10, 'FontWeight', 'bold', ...
                'ForegroundColor', [1 1 1], ...
                'BackgroundColor', [0.15 0.16 0.20], ...
                'BorderType', 'none', 'HighlightColor', [0.7 0.3 0.9]);
            sg4 = uigridlayout(sp4, [1 1]);
            sg4.Padding = [4 4 4 4];
            app.InfoText = uitextarea(sg4, ...
                'Value', '就绪。\n\n请选择场景并点击运行仿真，\n或点击加载数据打开已有结果。', ...
                'FontSize', 9, 'FontName', 'Microsoft YaHei', 'Editable', 'off', ...
                'FontColor', [0.8 0.8 0.8], 'BackgroundColor', [0.15 0.16 0.20]);

            % ===== 右侧：标签页 =====
            tabg = uitabgroup(grid);
            tabg.Layout.Row = 1;
            tabg.Layout.Column = 2;

            % 标签页 1: 3D 场景
            tab1 = uitab(tabg, 'Title', '  3D 场景  ');
            t1g = uigridlayout(tab1, [1 1]);
            t1g.Padding = [0 0 0 0];
            app.Axes3D = uiaxes(t1g);
            app.Axes3D.Color = [0.12 0.12 0.15];
            app.Axes3D.Box = 'on';
            app.Axes3D.XColor = [1 1 1]; app.Axes3D.YColor = [1 1 1]; app.Axes3D.ZColor = [1 1 1];
            app.Axes3D.Title.Color = [1 1 1];
            title(app.Axes3D, '请加载数据', 'FontSize', 12);

            % 标签页 2: 波形 (子页)
            tab2 = uitab(tabg, 'Title', '  波形  ');
            t2_wrap = uigridlayout(tab2, [2 1]);
            t2_wrap.RowHeight = {28, '1x'};
            t2_wrap.Padding = [4 2 2 4];
            t2_wrap.RowSpacing = 2;
            % 通道选择
            app.WaveChannel = uidropdown(t2_wrap, ...
                'Items', {'左右耳都显示', '只显示左耳', '只显示右耳'}, ...
                'Value', '左右耳都显示', 'FontSize', 9, ...
                'ValueChangedFcn', @(~,~) on_wave_channel(app));
            sub_tg = uitabgroup(t2_wrap);
            % 子页 2a: HRTF 处理后
            sub1 = uitab(sub_tg, 'Title', '  HRTF处理后  ');
            s1g = uigridlayout(sub1, [1 1]);
            s1g.Padding = [4 4 4 4];
            app.AxesWave = uiaxes(s1g);
            app.AxesWave.Color = [0.15 0.15 0.15];
            app.AxesWave.XColor = [1 1 1]; app.AxesWave.YColor = [1 1 1];
            app.AxesWave.Title.Color = [1 1 1];
            title(app.AxesWave, '请加载数据', 'FontSize', 12);
            % 子页 2b: HRTF 处理前 (原始房间声学信号)
            sub2 = uitab(sub_tg, 'Title', '  HRTF处理前  ');
            s2g = uigridlayout(sub2, [1 1]);
            s2g.Padding = [4 4 4 4];
            app.AxesWaveRaw = uiaxes(s2g);
            app.AxesWaveRaw.Color = [0.15 0.15 0.15];
            app.AxesWaveRaw.XColor = [1 1 1]; app.AxesWaveRaw.YColor = [1 1 1];
            app.AxesWaveRaw.Title.Color = [1 1 1];
            title(app.AxesWaveRaw, '请加载数据', 'FontSize', 12);

            % 标签页 3: ITD/ILD
            tab3 = uitab(tabg, 'Title', '  定位线索  ');
            t3g = uigridlayout(tab3, [2 1]);
            t3g.RowHeight = {'1x', 40};
            t3g.Padding = [4 4 2 4];
            t3g.RowSpacing = 2;
            app.AxesAnal = uiaxes(t3g);
            app.AxesAnal.Color = [0.15 0.15 0.15];
            app.AxesAnal.XColor = [1 1 1]; app.AxesAnal.YColor = [1 1 1];
            app.AxesAnal.Title.Color = [1 1 1];
            title(app.AxesAnal, '请加载数据', 'FontSize', 12);
            % ITD / ILD 缩放控制
            zoom_g = uigridlayout(t3g, [1 4]);
            zoom_g.ColumnWidth = {60, '1x', 60, '1x'};
            zoom_g.Padding = [2 0 2 0]; zoom_g.ColumnSpacing = 4;
            uilabel(zoom_g, 'Text', 'ITD±ms:', 'FontSize', 8, ...
                'FontColor', [0.1 0.6 0.25], 'HorizontalAlignment', 'right');
            app.ItdZoom = uislider(zoom_g, 'Limits', [0.1 3], 'Value', 1.2, ...
                'ValueChangedFcn', @(~,~) on_anal_zoom(app));
            uilabel(zoom_g, 'Text', 'ILD±dB:', 'FontSize', 8, ...
                'FontColor', [0.65 0.1 0.55], 'HorizontalAlignment', 'right');
            app.IldZoom = uislider(zoom_g, 'Limits', [1 30], 'Value', 15, ...
                'ValueChangedFcn', @(~,~) on_anal_zoom(app));

            % 标签页 4: 定位评估 (v3.0)
            tab4 = uitab(tabg, 'Title', '  定位评估  ');
            t4g = uigridlayout(tab4, [2 2]);
            t4g.RowHeight = {'1x', '1x'};
            t4g.ColumnWidth = {'1x', '1x'};
            t4g.Padding = [4 4 4 4];
            t4g.RowSpacing = 4;
            t4g.ColumnSpacing = 4;

            app.AxesEvalDOA = uiaxes(t4g);
            app.AxesEvalDOA.Layout.Row = 1;
            app.AxesEvalDOA.Layout.Column = 1;
            app.AxesEvalDOA.Color = [0.15 0.15 0.15];
            app.AxesEvalDOA.XColor = [1 1 1]; app.AxesEvalDOA.YColor = [1 1 1];
            app.AxesEvalDOA.Title.Color = [1 1 1];
            title(app.AxesEvalDOA, 'DOA 轨迹对比', 'FontSize', 11, 'FontWeight', 'bold');

            app.AxesEvalErr = uiaxes(t4g);
            app.AxesEvalErr.Layout.Row = 1;
            app.AxesEvalErr.Layout.Column = 2;
            app.AxesEvalErr.Color = [0.15 0.15 0.15];
            app.AxesEvalErr.XColor = [1 1 1]; app.AxesEvalErr.YColor = [1 1 1];
            app.AxesEvalErr.Title.Color = [1 1 1];
            title(app.AxesEvalErr, '角度误差分布', 'FontSize', 11, 'FontWeight', 'bold');

            app.AxesEvalCM = uiaxes(t4g);
            app.AxesEvalCM.Layout.Row = 2;
            app.AxesEvalCM.Layout.Column = 1;
            app.AxesEvalCM.Color = [0.15 0.15 0.15];
            app.AxesEvalCM.XColor = [1 1 1]; app.AxesEvalCM.YColor = [1 1 1];
            app.AxesEvalCM.Title.Color = [1 1 1];
            title(app.AxesEvalCM, '混淆矩阵 (前-右/前-左/后-右/后-左)', 'FontSize', 11, 'FontWeight', 'bold');

            ep = uipanel(t4g, 'Title', ' 评估指标 ', ...
                'FontSize', 10, 'FontWeight', 'bold', ...
                'BackgroundColor', [0.98 0.98 0.98]);
            ep.Layout.Row = 2;
            ep.Layout.Column = 2;
            ep_g = uigridlayout(ep, [1 1]);
            ep_g.Padding = [4 4 4 4];
            app.EvalText = uitextarea(ep_g, ...
                'Value', '请先加载 v3.0 仿真结果 (.mat 含定位评估字段)', ...
                'FontSize', 10, 'FontName', 'Consolas', 'Editable', 'off', ...
                'FontColor', [0 0 0], 'BackgroundColor', [1 1 1]);

% ===== 状态栏 =====
            app.StatusBar = uilabel(grid, ...
                'Text', '就绪', 'FontSize', 10, 'FontColor', [0.8 0.8 0.8], ...
                'BackgroundColor', [0.15 0.16 0.20], ...
                'HorizontalAlignment', 'left');
            app.StatusBar.Layout.Row = 2;
            app.StatusBar.Layout.Column = [1 2];

            app.ProgressBar = uilabel(grid, ...
                'Text', '', 'Visible', 'off', 'FontSize', 9, ...
                'FontColor', [0.2 0.5 0.8], ...
                'BackgroundColor', [0.92 0.92 0.92], ...
                'HorizontalAlignment', 'right');
            app.ProgressBar.Layout.Row = 2;
            app.ProgressBar.Layout.Column = [1 2];
        end
    end

    % ============ 初始化 ============
    methods (Access = private)
        function startupFcn(app)
            try
                % 尝试多种方式定位项目根目录
                candidates = {};
                try
                    mf = mfilename('fullpath');
                    if ~isempty(mf)
                        candidates{end+1} = fileparts(fileparts(mf));
                    end
                catch
                end
                try
                    w = which('binaural_gui');
                    if ~isempty(w)
                        candidates{end+1} = fileparts(fileparts(w));
                    end
                catch
                end
                candidates{end+1} = 'D:\shengxuedingwei2';

                app.proj_root = '';
                for i = 1:length(candidates)
                    if exist(fullfile(candidates{i}, 'scenes'), 'dir')
                        app.proj_root = candidates{i};
                        break;
                    end
                end
                if isempty(app.proj_root)
                    app.proj_root = 'D:\shengxuedingwei2';
                end

                app.results_dir = fullfile(app.proj_root, 'results');
                if ~exist(app.results_dir, 'dir')
                    try mkdir(app.results_dir); catch; end
                end
                % 加载头部 3D 模型
                app.try_load_head();
                refresh_scenes(app);
                status(app, '就绪。请选择场景或加载数据文件。');
            catch e
                % startupFcn 绝对不能抛异常，否则 app 处于半析构状态
                try
                    app.proj_root = 'D:\shengxuedingwei2';
                    app.results_dir = fullfile(app.proj_root, 'results');
                catch
                end
            end
        end

        function refresh_scenes(app)
            if ~isvalid(app), return; end
            files = dir(fullfile(app.proj_root, 'scenes', '*.yaml'));
            if isempty(files)
                app.scene_list = struct('name', {}, 'path', {});
                app.SceneDrop.Items = {'<未找到场景文件>'};
            else
                n = length(files);
                app.scene_list = struct('name', cell(1,n), 'path', cell(1,n));
                for i = 1:n
                    app.scene_list(i).name = files(i).name;
                    app.scene_list(i).path = fullfile(files(i).folder, files(i).name);
                end
                app.SceneDrop.Items = [{'(请选择场景)'}, {files.name}];
            end
        end

        function on_scene_select(app)
            if ~isvalid(app), return; end
            idx = app.SceneDrop.ValueIndex;
            if idx <= 1, return; end
            yaml_path = app.scene_list(idx-1).path;
            preview = parse_yaml_preview(yaml_path);
            if isempty(preview), return; end
            render_preview(app, preview);
            status(app, ['已选择: ' app.scene_list(idx-1).name], [0 0.3 0]);
        end

        function try_load_head(app)
            if app.head_loaded, return; end
            candidates = { ...
                fullfile(app.proj_root, 'matlab', 'MaleHead.obj'), ...
                'D:\shengxuedingwei2\matlab\MaleHead.obj', ...
                'D:\shengxuedingwei\config\microphone\MaleHead.obj' ...
            };
            for c = 1:length(candidates)
                p = candidates{c};
                if ~exist(p, 'file'), continue; end
                % 内联 OBJ 解析 (已验证可用的逻辑)
                fid = fopen(p, 'r');
                if fid == -1, continue; end
                vtxList = []; faceList = {};
                while ~feof(fid)
                    ln = fgetl(fid);
                    if ~ischar(ln), break; end
                    if isempty(ln) || ln(1) == '#', continue; end
                    parts = strsplit(strtrim(ln));
                    if isempty(parts), continue; end
                    if strcmp(parts{1}, 'v') && length(parts) >= 4
                        vtxList(end+1,:) = [str2double(parts{2}), str2double(parts{3}), str2double(parts{4})];
                    elseif strcmp(parts{1}, 'f')
                        fv = [];
                        for i = 2:length(parts)
                            ip = strsplit(parts{i}, '/');
                            idx = str2double(ip{1});
                            if idx < 0, idx = size(vtxList,1) + idx + 1; end
                            fv(end+1) = idx;
                        end
                        if ~isempty(fv), faceList{end+1} = fv; end
                    end
                end
                fclose(fid);
                if isempty(vtxList), continue; end
                fcs = [];
                for i = 1:length(faceList)
                    fv = faceList{i};
                    if length(fv) == 3
                        fcs(end+1,:) = fv;
                    elseif length(fv) > 3
                        for j = 2:length(fv)-1
                            fcs(end+1,:) = [fv(1), fv(j), fv(j+1)];
                        end
                    end
                end
                if ~isempty(fcs)
                    vtxList = vtxList - mean(vtxList);
                    ext = max(vtxList) - min(vtxList);
                    scl = max(ext) / 2;
                    if scl < 0.01, scl = 0.09; end
                    app.head_model = struct('vertices', vtxList, 'faces', fcs, 'scale', scl);
                    app.head_loaded = true;
                    return;
                end
            end
        end

        function status(app, msg, color)
            if nargin < 3, color = [0.3 0.3 0.3]; end
            app.StatusBar.Text = msg;
            app.StatusBar.FontColor = color;
            drawnow;
        end

        function progress(app, pct)
            if pct <= 0
                app.ProgressBar.Visible = 'off';
                return
            end
            n = max(1, round(pct / 5));
            bar_str = [char(ones(1,n)*char(9608)) char(ones(1,20-n)*char(9617))];
            app.ProgressBar.Text = sprintf('仿真中... %s %d%%', bar_str, round(pct));
            app.ProgressBar.Visible = 'on';
            drawnow;
        end
    end

    % ============ 仿真 ============
    methods (Access = private)
        function on_run(app, ~)
            if ~isvalid(app) || ~isvalid(app.SceneDrop), return; end
            try
                idx = app.SceneDrop.ValueIndex;
            catch
                return;
            end
            if idx <= 1
                status(app, '请先选择场景。', [0.8 0.3 0]);
                return;
            end
            scene_file = app.scene_list(idx-1).path;
            status(app, '正在运行仿真...', [0 0 0.6]);
            progress(app, 10);
            app.RunBtn.Enable = 'off';
            drawnow;

            % 提取 HRTF 模式 (取第一个词: analytical / parametric / measured)
            hrtf_val = app.HrtfDrop.Value;
            hrtf_mode = strtok(hrtf_val);
            cmd = sprintf('cd /d "%s" && python run.py "%s" --no-viz --hrtf %s', ...
                app.proj_root, scene_file, hrtf_mode);
            % Append source override if selected
            src_val = app.SourceDrop.Value;
            if ~startsWith(src_val, '(')
                cmd = [cmd ' --source ' src_val];
            end
            [exit_code, result] = system(cmd);

            app.RunBtn.Enable = 'on';
            progress(app, 0);

            if exit_code == 0
                status(app, '仿真完成。', [0 0.5 0]);
                files = dir(fullfile(app.results_dir, '*.mat'));
                if ~isempty(files)
                    [~, si] = sort([files.datenum], 'descend');
                    load_mat(app, fullfile(app.results_dir, files(si(1)).name));
                end
            else
                status(app, ['错误: ' strtrim(result(1:min(100,end)))], [0.8 0 0]);
            end
        end
    end

    % ============ 数据加载 ============
    methods (Access = private)
        function on_load(app, ~)
            if ~isvalid(app), return; end
            if isempty(app.results_dir) || ~exist(app.results_dir, 'dir')
                app.results_dir = 'D:\shengxuedingwei2\results';
            end
            [file, folder] = uigetfile(fullfile(app.results_dir, '*.mat'), ...
                '加载仿真结果');
            if isequal(file, 0), return; end
            load_mat(app, fullfile(folder, file));
        end

        function load_mat(app, fpath)
            stop_anim(app);
            try
                raw = load(fpath);
            catch
                status(app, ['加载失败: ' fpath], [0.8 0 0]);
                return;
            end

            required = {'stereo_signal','trajectory','fs','room_dims', ...
                        'head_center','left_ear','right_ear','scene_name'};
            for k = 1:length(required)
                if ~isfield(raw, required{k})
                    status(app, ['缺少字段: ' required{k}], [0.8 0 0]);
                    return;
                end
            end

            for f = {'scene_name','timestamp','source_generator','motion_type'}
                if isfield(raw,f{1}) && iscell(raw.(f{1}))
                    raw.(f{1}) = raw.(f{1}){1};
                end
            end

            raw.fs = double(raw.fs(1));
            if isfield(raw,'absorption'), raw.absorption = double(raw.absorption(1)); end
            if isfield(raw,'max_order'), raw.max_order = double(raw.max_order(1)); end
            if isfield(raw,'head_radius'), raw.head_radius = double(raw.head_radius(1)); end
            if isfield(raw,'source_duration'), raw.source_duration = double(raw.source_duration(1)); end
            if isfield(raw,'motion_enabled'), raw.motion_enabled = double(raw.motion_enabled(1)); end

            app.data = raw;
            app.anim_idx = 1;

            % v3.0: 解析定位评估数据
            if isfield(raw, 'doa_estimated')
                app.doa_estimated = raw.doa_estimated(:)';
            else
                app.doa_estimated = [];
            end
            if isfield(raw, 'doa_truth')
                app.doa_truth = raw.doa_truth(:)';
            else
                app.doa_truth = [];
            end
            if isfield(raw, 'loc_timestamps')
                app.loc_timestamps = raw.loc_timestamps(:)';
            else
                app.loc_timestamps = [];
            end
            % 构建 eval_metrics 结构
            app.eval_metrics = struct();
            if isfield(raw, 'eval_rmse'), app.eval_metrics.rmse = double(raw.eval_rmse(1)); end
            if isfield(raw, 'eval_mae'), app.eval_metrics.mae = double(raw.eval_mae(1)); end
            if isfield(raw, 'eval_accuracy_15deg'), app.eval_metrics.accuracy_15deg = double(raw.eval_accuracy_15deg(1)); end
            if isfield(raw, 'eval_front_back_conf'), app.eval_metrics.fb_confusion = double(raw.eval_front_back_conf(1)); end
            if isfield(raw, 'eval_confusion_matrix'), app.eval_metrics.cm = raw.eval_confusion_matrix; end
            % ITD/ILD only DOA for comparison tab
            if isfield(raw, 'doa_itd_only'), app.doa_itd_only = raw.doa_itd_only(:)'; end
            if isfield(raw, 'doa_ild_only'), app.doa_ild_only = raw.doa_ild_only(:)'; end

            % 提取原始信号（HRTF处理前）
            if isfield(raw, 'stereo_raw')
                app.data.stereo_raw = raw.stereo_raw;
            end

            render_3d(app);
            render_wave(app);
            render_wave_raw(app);
            render_anal(app);
            render_eval(app);
            update_info(app);

            [~,fname] = fileparts(fpath);
            status(app, sprintf('已加载: %s  |  %s  |  %.2f 秒', ...
                fname, app.data.scene_name, ...
                size(app.data.stereo_signal,1)/app.data.fs), [0 0.4 0]);
        end
    end

    % ============ 场景预览 (选场景即显示) ============
    methods (Access = private)
        function render_preview(app, preview)
            ax = app.Axes3D;
            cla(ax); hold(ax, 'on');
            x = preview.room_dims(1); y = preview.room_dims(2); z = preview.room_dims(3);

            % 房间墙壁
            v = [0 0 0; x 0 0; x y 0; 0 y 0; 0 0 z; x 0 z; x y z; 0 y z];
            face_idx = {[1 2 3 4], [5 6 7 8], [1 2 6 5], [2 3 7 6], [3 4 8 7], [4 1 5 8]};
            wall_colors = {[0.65 0.78 0.92], [0.70 0.82 0.94], ...
                           [0.72 0.84 0.95], [0.74 0.85 0.96], ...
                           [0.70 0.82 0.94], [0.68 0.80 0.93]};
            for i = 1:6
                patch(ax, v(face_idx{i},1), v(face_idx{i},2), v(face_idx{i},3), ...
                    wall_colors{i}, 'FaceAlpha', 0.12, 'EdgeColor', [0.5 0.5 0.55], ...
                    'LineWidth', 0.8);
            end

            % 地面网格
            [gx, gy] = meshgrid(0:1:x, 0:1:y);
            scatter3(ax, gx(:), gy(:), zeros(size(gx(:))), 4, [0.6 0.6 0.6], 'o', ...
                'MarkerFaceAlpha', 0.25, 'MarkerEdgeAlpha', 0);

            % 头部 3D 模型 — 直接内联加载渲染
            hc = preview.head_center; hr = preview.head_radius;
            [hv, hf] = load_head_model();
            if ~isempty(hv)
                patch(ax, 'Faces', hf, 'Vertices', hv*(hr/0.09)+hc, ...
                    'FaceColor', [0.85 0.82 0.78], 'FaceAlpha', 0.7, ...
                    'EdgeColor', 'none', 'FaceLighting', 'gouraud');
            else
                [sx, sy, sz] = sphere(24);
                surf(ax, sx*hr+hc(1), sy*hr+hc(2), sz*hr+hc(3), ...
                    'FaceColor', [0.85 0.82 0.78], 'FaceAlpha', 0.6, ...
                    'EdgeColor', 'none', 'FaceLighting', 'gouraud');
            end

            % 双耳
            le = preview.left_ear; re = preview.right_ear;
            scatter3(ax, le(1), le(2), le(3), ...
                150, [0.1 0.35 0.9], '^', 'filled', 'MarkerEdgeColor', 'k');
            scatter3(ax, re(1), re(2), re(3), ...
                150, [0.9 0.2 0.2], 'v', 'filled', 'MarkerEdgeColor', 'k');

            % 头部朝向线 (preview: yaw from config)
            fwd_len = preview.head_radius * 1.5;
            yaw_h = 0;
            if isfield(preview, 'head_yaw') && ~isempty(preview.head_yaw)
                yaw_h = deg2rad(preview.head_yaw);
            end
            plot3(ax, [hc(1) hc(1)+fwd_len*sin(yaw_h)], ...
                      [hc(2) hc(2)+fwd_len*cos(yaw_h)], ...
                      [hc(3) hc(3)], 'y-', 'LineWidth', 2.5);

            % 运动轨迹预览
            if isfield(preview, 'traj_preview') && ~isempty(preview.traj_preview)
                tp = preview.traj_preview;
                plot3(ax, tp(:,1), tp(:,2), tp(:,3), '--', ...
                    'Color', [0.15 0.45 0.75], 'LineWidth', 1.2);
                scatter3(ax, tp(1,1), tp(1,2), tp(1,3), ...
                    80, [0.2 0.75 0.25], 'o', 'filled', 'MarkerEdgeColor', 'k');
            end

            % 光照
            light(ax, 'Position', [x/2 y/2 z*2], 'Style', 'local');
            lighting(ax, 'gouraud');

            xlabel(ax, 'X (m)'); ylabel(ax, 'Y (m)'); zlabel(ax, 'Z (m)');
            title(ax, ['场景预览: ' strrep(preview.name, '_', '\_')], ...
                'FontSize', 14, 'FontWeight', 'bold');
            xlim(ax, [0 x]); ylim(ax, [0 y]); zlim(ax, [0 z]);
            axis(ax, 'equal');
            view(ax, 50, 28);
            rotate3d(ax, 'on');
            grid(ax, 'on'); ax.GridAlpha = 0.2;
            hold(ax, 'off');
        end
    end

    % ============ 3D 场景渲染 ============
    methods (Access = private)
        function render_3d(app)
            ax = app.Axes3D;
            cla(ax); hold(ax, 'on');
            d = app.data;
            if isempty(d), return; end

            x = d.room_dims(1); y = d.room_dims(2); z = d.room_dims(3);

            % 房间墙壁
            v = [0 0 0; x 0 0; x y 0; 0 y 0; 0 0 z; x 0 z; x y z; 0 y z];
            face_idx = {[1 2 3 4], [5 6 7 8], [1 2 6 5], [2 3 7 6], [3 4 8 7], [4 1 5 8]};
            wall_colors = {[0.65 0.78 0.92], [0.70 0.82 0.94], ...
                           [0.72 0.84 0.95], [0.74 0.85 0.96], ...
                           [0.70 0.82 0.94], [0.68 0.80 0.93]};
            for i = 1:6
                patch(ax, v(face_idx{i},1), v(face_idx{i},2), v(face_idx{i},3), ...
                    wall_colors{i}, 'FaceAlpha', 0.12, 'EdgeColor', [0.5 0.5 0.55], ...
                    'LineWidth', 0.8);
            end

            % 地面网格
            [gx, gy] = meshgrid(0:1:x, 0:1:y);
            scatter3(ax, gx(:), gy(:), zeros(size(gx(:))), 4, [0.6 0.6 0.6], 'o', ...
                'MarkerFaceAlpha', 0.25, 'MarkerEdgeAlpha', 0);

            % 头部 3D 模型
            hc = d.head_center; hr = d.head_radius;
            [hv, hf] = load_head_model();
            if ~isempty(hv)
                patch(ax, 'Faces', hf, 'Vertices', hv*(hr/0.09)+hc, ...
                    'FaceColor', [0.85 0.82 0.78], 'FaceAlpha', 0.7, ...
                    'EdgeColor', 'none', 'FaceLighting', 'gouraud');
            else
                [sx, sy, sz] = sphere(24);
                surf(ax, sx*hr+hc(1), sy*hr+hc(2), sz*hr+hc(3), ...
                    'FaceColor', [0.85 0.82 0.78], 'FaceAlpha', 0.6, ...
                    'EdgeColor', 'none', 'FaceLighting', 'gouraud');
            end

            % 双耳位置
            scatter3(ax, d.left_ear(1), d.left_ear(2), d.left_ear(3), ...
                150, [0.1 0.35 0.9], '^', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1);
            scatter3(ax, d.right_ear(1), d.right_ear(2), d.right_ear(3), ...
                150, [0.9 0.2 0.2], 'v', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1);

            % 头部朝向线 (根据 yaw 旋转)
            hc = d.head_center; hr = d.head_radius;
            yaw_h = 0;
            if isfield(d, 'head_yaw_deg')
                ts_idx = min(app.anim_idx, length(d.head_yaw_deg));
                yaw_h = deg2rad(d.head_yaw_deg(ts_idx));
            end
            fwd_len = hr * 1.5;
            dx = fwd_len * sin(yaw_h);
            dy = fwd_len * cos(yaw_h);
            app.h_head_dir = plot3(ax, [hc(1) hc(1)+dx], ...
                [hc(2) hc(2)+dy], [hc(3) hc(3)], 'y-', 'LineWidth', 2.5);

            % 运动轨迹
            traj = d.trajectory;
            if size(traj,1) > 1 && size(traj,2) >= 3
                stride = max(1, floor(size(traj,1) / 600));
                plot3(ax, traj(1:stride:end,1), traj(1:stride:end,2), ...
                    traj(1:stride:end,3), '-', 'Color', [0.15 0.45 0.75], 'LineWidth', 1.6);
                scatter3(ax, traj(1,1), traj(1,2), traj(1,3), ...
                    80, [0.2 0.75 0.25], 'o', 'filled', 'MarkerEdgeColor', 'k');
                scatter3(ax, traj(end,1), traj(end,2), traj(end,3), ...
                    80, [0.85 0.25 0.15], 's', 'filled', 'MarkerEdgeColor', 'k');
            end

            % 声源（动画点）
            app.h_source = scatter3(ax, traj(1,1), traj(1,2), traj(1,3), ...
                200, [1.0 0.5 0.0], 'o', 'filled', ...
                'MarkerEdgeColor', 'k', 'LineWidth', 2);

            % 投影线
            app.h_cursor3d = plot3(ax, [traj(1,1) traj(1,1)], ...
                [traj(1,2) traj(1,2)], [0 z], '--', ...
                'Color', [0.9 0.4 0.1], 'LineWidth', 1);

            % 信息浮层
            app.h_overlay = text(ax, x*0.98, y*0.95, z*0.95, ...
                overlay_str(app, traj(1,:)), ...
                'FontSize', 9, 'FontName', 'Consolas', ...
                'BackgroundColor', [0 0 0 0.6], 'Color', 'white', ...
                'EdgeColor', 'none', 'HorizontalAlignment', 'right', ...
                'VerticalAlignment', 'top');

            % 光照
            light(ax, 'Position', [x/2 y/2 z*2], 'Style', 'local');
            lighting(ax, 'gouraud');

            xlabel(ax, 'X (m)'); ylabel(ax, 'Y (m)'); zlabel(ax, 'Z (m)');
            title(ax, strrep(d.scene_name, '_', '\_'), 'FontSize', 14, 'FontWeight', 'bold');
            xlim(ax, [0 x]); ylim(ax, [0 y]); zlim(ax, [0 z]);
            axis(ax, 'equal');
            view(ax, 50, 28);
            rotate3d(ax, 'on');
            grid(ax, 'on');
            ax.GridAlpha = 0.2;
            hold(ax, 'off');
        end

        function s = overlay_str(~, pos, d)
            persistent data_ref;
            if nargin == 3, data_ref = d; end
            s = sprintf('[%.2f, %.2f, %.2f]', pos(1), pos(2), pos(3));
        end

        function update_overlay(app, pos)
            d = app.data;
            dist_l = norm(pos - d.left_ear');
            dist_r = norm(pos - d.right_ear');
            t_cur = (app.anim_idx - 1) / d.fs;
            app.h_overlay.String = sprintf(...
                '时刻: %.2f s  |  [%.2f, %.2f, %.2f]  |  左耳: %.2f m  右耳: %.2f m', ...
                t_cur, pos(1), pos(2), pos(3), dist_l, dist_r);
        end
    end

    % ============ 波形渲染 ============
    methods (Access = private)
        function render_wave(app)
            ax = app.AxesWave;
            cla(ax); hold(ax, 'on');
            d = app.data;
            if isempty(d), return; end

            s = d.stereo_signal;
            t = (0:size(s,1)-1)' / d.fs;
            ch = app.WaveChannel.Value;
            show_l = contains(ch, '左') || contains(ch, '都');
            show_r = contains(ch, '右') || contains(ch, '都');
            if show_l
                plot(ax, t, s(:,1), 'Color', [0.1 0.35 0.9], 'LineWidth', 0.6);
            end
            if show_r
                plot(ax, t, s(:,2), 'Color', [0.9 0.2 0.2], 'LineWidth', 0.6);
            end
            xlim(ax, [0 t(end)]); ylim(ax, [-1.05 1.05]);
            xlabel(ax, '时间 (s)'); ylabel(ax, '幅值');
            title(ax, ['双耳波形  —  ' ch], 'FontSize', 12, 'FontWeight', 'bold');
            if show_l && show_r
                legend(ax, {'左耳', '右耳'}, 'Location', 'northeast', 'FontSize', 9);
            end
            grid(ax, 'on'); ax.GridAlpha = 0.25;
            hold(ax, 'off');
        end

        function render_wave_raw(app)
            ax = app.AxesWaveRaw;
            cla(ax); hold(ax, 'on');
            d = app.data;
            if isempty(d), return; end
            if ~isfield(d, 'stereo_raw')
                title(ax, 'HRTF处理前 — 无数据 (需重新运行仿真)', 'FontSize', 11);
                hold(ax, 'off'); return;
            end
            s = d.stereo_raw;
            t = (0:size(s,1)-1)' / d.fs;
            ch = app.WaveChannel.Value;
            show_l = contains(ch, '左') || contains(ch, '都');
            show_r = contains(ch, '右') || contains(ch, '都');
            if show_l
                plot(ax, t, s(:,1), 'Color', [0.1 0.35 0.9], 'LineWidth', 0.6);
            end
            if show_r
                plot(ax, t, s(:,2), 'Color', [0.9 0.2 0.2], 'LineWidth', 0.6);
            end
            xlim(ax, [0 t(end)]); ylim(ax, [-1.05 1.05]);
            xlabel(ax, '时间 (s)'); ylabel(ax, '幅值');
            title(ax, ['原始房间声学信号 — HRTF处理前  —  ' ch], ...
                'FontSize', 12, 'FontWeight', 'bold');
            if show_l && show_r
                legend(ax, {'左耳', '右耳'}, 'Location', 'northeast', 'FontSize', 9);
            end
            grid(ax, 'on'); ax.GridAlpha = 0.25;
            hold(ax, 'off');
        end
    end

    % ============ ITD/ILD 分析 ============
    methods (Access = private)
        function render_anal(app)
            ax = app.AxesAnal;
            cla(ax);
            d = app.data;
            if isempty(d), return; end

            fs = d.fs;
            s = d.stereo_signal;
            n = size(s, 1);

            win_len = round(0.05 * fs);
            hop = max(1, round(win_len / 2));
            n_win = min(150, floor((n - win_len) / hop) + 1);
            max_lag = round(0.001 * fs);

            % FFT size for GCC-PHAT
            nfft = 2^nextpow2(2 * win_len);

            itd = zeros(1, n_win); ild = zeros(1, n_win); tv = zeros(1, n_win);
            for i = 1:n_win
                w0 = (i-1)*hop + 1; w1 = min(w0+win_len-1, n);
                seg_l = s(w0:w1,1); seg_r = s(w0:w1,2);

                % ILD: energy ratio (always stable for any signal)
                ild(i) = 10 * log10((sum(seg_l.^2)+1e-10) / (sum(seg_r.^2)+1e-10));

                % GCC-PHAT for robust ITD
                L = fft(seg_l .* hanning(win_len), nfft);
                R = fft(seg_r .* hanning(win_len), nfft);
                X = L .* conj(R);
                X_phat = X ./ (abs(X) + 1e-10);
                gcc = real(ifft(X_phat));

                % Search peak in valid ITD range
                search_region = [gcc(end-max_lag+1:end); gcc(1:max_lag+1)];
                [~, pk] = max(search_region);
                lag = pk - max_lag - 1;
                itd(i) = lag / fs * 1000;  % ms, positive=right leads
                tv(i) = (w0+w1)/2/fs;
            end

            app.itd_cache = itd; app.ild_cache = ild; app.t_analysis = tv;

            yyaxis(ax, 'left');
            plot(ax, tv, itd, '-', 'Color', [0.1 0.6 0.25], 'LineWidth', 1.5);
            ylabel(ax, 'ITD (ms)');
            itd_r = app.ItdZoom.Value;
            ylim(ax, [-itd_r itd_r]);

            yyaxis(ax, 'right');
            plot(ax, tv, ild, '-', 'Color', [0.65 0.1 0.55], 'LineWidth', 1.5);
            ylabel(ax, 'ILD (dB)');
            ild_r = app.IldZoom.Value;
            ylim(ax, [-ild_r ild_r]);

            xlim(ax, [0 n/fs]);
            xlabel(ax, '时间 (s)');
            title(ax, 'ITD (绿色) / ILD (紫色) — 双耳定位线索随时间变化', ...
                'FontSize', 12, 'FontWeight', 'bold');
            grid(ax, 'on'); ax.GridAlpha = 0.25;
        end
    end

    % ============ 定位评估渲染 (v3.0) ============
    methods (Access = private)
        function render_eval(app)
            % DOA 轨迹对比
            ax1 = app.AxesEvalDOA;
            cla(ax1); hold(ax1, 'on');

            if isempty(app.doa_estimated) || isempty(app.doa_truth)
                title(ax1, 'DOA 轨迹对比 — 无评估数据', 'FontSize', 11);
                hold(ax1, 'off'); return;
            end

            t = app.loc_timestamps;
            plot(ax1, t, app.doa_truth, 'w-', 'LineWidth', 1.5);
            plot(ax1, t, app.doa_estimated, 'r-', 'LineWidth', 1.2);
            xlabel(ax1, '时间 (s)');
            ylabel(ax1, '方位角 (deg)');
            legend(ax1, {'真实 DOA', '估计 DOA'}, 'Location', 'best', 'FontSize', 8);
            ylim(ax1, [-180 180]);
            grid(ax1, 'on'); ax1.GridAlpha = 0.2;
            title(ax1, 'DOA 轨迹对比 — 黑:真实  红:估计', 'FontSize', 11, 'FontWeight', 'bold');
            hold(ax1, 'off');

            % 角度误差分布
            ax2 = app.AxesEvalErr;
            cla(ax2);
            errors = app.doa_estimated - app.doa_truth;
            errors = mod(errors + 180, 360) - 180;
            histogram(ax2, errors, 36, 'FaceColor', [0.275 0.51 0.71], ...
                'EdgeColor', 'white');
            xline(ax2, 0, 'k-', 'LineWidth', 1);
            xline(ax2, mean(errors), 'r--', 'LineWidth', 1);
            xlabel(ax2, '角度误差 (deg)');
            ylabel(ax2, '帧数');
            mean_str = sprintf('均值=%.1f deg', mean(errors));
            legend(ax2, {mean_str}, 'Location', 'northeast', 'FontSize', 8);
            grid(ax2, 'on'); ax2.GridAlpha = 0.2;
            title(ax2, '角度误差分布直方图', 'FontSize', 11, 'FontWeight', 'bold');

            % 混淆矩阵
            ax3 = app.AxesEvalCM;
            cla(ax3);
            if isfield(app.eval_metrics, 'cm') && ~isempty(app.eval_metrics.cm)
                cm = app.eval_metrics.cm;
                imagesc(ax3, cm);
                colormap(ax3, hot);
                colorbar(ax3);
                labels = {'前-右', '前-左', '后-右', '后-左'};
                ax3.XTick = 1:4; ax3.YTick = 1:4;
                ax3.XTickLabel = labels; ax3.YTickLabel = labels;
                ax3.XAxis.FontSize = 8; ax3.YAxis.FontSize = 8;
                for i = 1:4
                    for j = 1:4
                        if cm(i,j) > 0
                            c = 'k'; if cm(i,j) > max(cm(:))/2, c = 'w'; end
                            text(ax3, j, i, num2str(cm(i,j)), ...
                                'HorizontalAlignment', 'center', ...
                                'Color', c, 'FontSize', 10, 'FontWeight', 'bold');
                        end
                    end
                end
                xlabel(ax3, '估计'); ylabel(ax3, '真实');
                title(ax3, 'DOA 象限混淆矩阵', 'FontSize', 11, 'FontWeight', 'bold');
            else
                title(ax3, '混淆矩阵 — 无数据', 'FontSize', 11);
            end

            % 评估指标文本
            nfr = length(app.doa_estimated);
            rmse_val = 0; mae_val = 0; acc10 = 0; fb = 0;
            if isfield(app.eval_metrics, 'rmse'), rmse_val = app.eval_metrics.rmse; end
            if isfield(app.eval_metrics, 'mae'), mae_val = app.eval_metrics.mae; end
            if isfield(app.eval_metrics, 'accuracy_15deg'), acc15 = app.eval_metrics.accuracy_15deg * 100; end
            if isfield(app.eval_metrics, 'fb_confusion'), fb = app.eval_metrics.fb_confusion * 100; end

            app.EvalText.Value = sprintf([...
                '评估帧数:         %d\n' ...
                'RMSE:             %.2f deg\n' ...
                'MAE:              %.2f deg\n' ...
                '准确率(15deg内):   %.1f %%\n' ...
                '前后混淆率:        %.1f %%\n\n' ...
                '定位算法:         %s'], ...
                nfr, rmse_val, mae_val, acc15, fb, ...
                app.data.loc_method);
        end
    end

    % ============ 定位诊断 (v3.0) ============
    methods (Access = private)

        function update_info(app)
            if ~isvalid(app), return; end
            d = app.data;
            if isempty(d), return; end
            n = size(d.stereo_signal, 1);
            abs_val = 0.3; max_ord = 8; hr = 0.09; src_dur = 0;
            if isfield(d,'absorption'), abs_val = d.absorption; end
            if isfield(d,'max_order'), max_ord = d.max_order; end
            if isfield(d,'head_radius'), hr = d.head_radius; end
            if isfield(d,'source_duration'), src_dur = d.source_duration; end
            gen = '未知'; if isfield(d,'source_generator'), gen = d.source_generator; end
            mot = '静态'; men = 0; if isfield(d,'motion_type'), mot = d.motion_type; end
            if isfield(d,'motion_enabled'), men = d.motion_enabled; end

            % 翻译运动类型
            mot_cn = mot;
            switch mot
                case 'static',  mot_cn = '静态';
                case 'linear',  mot_cn = '直线';
                case 'semicircle', mot_cn = '半圆';
                case 'circle',  mot_cn = '圆形';
            end

            % 翻译声源类型
            gen_cn = gen;
            switch gen
                case 'bowl_impact',  gen_cn = '碗撞击声';
                case 'chair_sliding', gen_cn = '椅子滑动声';
                case 'human_voice',  gen_cn = '人声';
                case 'sine',  gen_cn = '正弦波';
                case 'chirp', gen_cn = '扫频信号';
                case 'noise', gen_cn = '白噪声';
            end

            app.InfoText.Value = sprintf([...
                '场景:       %s\n' ...
                '房间:       %.1f x %.1f x %.1f m   吸收系数: %.2f   反射阶数: %d\n' ...
                '头部:       [%.2f, %.2f, %.2f]   半径: %.2f m\n' ...
                '左耳:       [%.2f, %.2f, %.2f]\n' ...
                '右耳:       [%.2f, %.2f, %.2f]\n' ...
                '声源:       %s   时长: %.2f s\n' ...
                '运动:       %s  (启用=%d)\n' ...
                '输出:       %d Hz   %d 采样   %.2f s 立体声'], ...
                d.scene_name, ...
                d.room_dims(1),d.room_dims(2),d.room_dims(3), abs_val, max_ord, ...
                d.head_center, hr, ...
                d.left_ear, d.right_ear, ...
                gen_cn, src_dur, ...
                mot_cn, men, ...
                d.fs, n, n/d.fs);
        end
    end

    % ============ 动画控制 ============
    methods (Access = private)
        function on_play_pause(app, ~)
            if ~isvalid(app), return; end
            if isempty(app.data)
                status(app, '请先加载数据文件。', [0.8 0.3 0]);
                return;
            end
            if app.is_playing
                stop_anim(app);
            else
                start_anim(app);
            end
        end

        function start_anim(app)
            app.is_playing = true;
            app.PlayBtn.Text = '暂停';
            app.PlayBtn.BackgroundColor = [1.0 0.55 0.1];
            t = timer('ExecutionMode', 'fixedRate', 'Period', 0.035, ...
                'BusyMode', 'drop', 'TimerFcn', @(~,~) anim_step(app));
            start(t);
            app.h_wave_cur = []; app.h_anal_cur = [];
            if isvalid(app.ProgressSlider)
                app.ProgressSlider.Limits = [0 1];
            end
        end

        function stop_anim(app)
            if ~isvalid(app), return; end
            try app.is_playing = false; catch; end
            try app.PlayBtn.Text = '播放'; catch; end
            try app.PlayBtn.BackgroundColor = [0.15 0.70 0.25]; catch; end
            ts = timerfindall;
            for i = 1:length(ts)
                try stop(ts(i)); delete(ts(i)); catch; end
            end
        end

                function anim_step(app)
            if ~isvalid(app), return; end
            if ~app.is_playing || isempty(app.data), return; end
            d = app.data;
            traj = d.trajectory;
            n = size(traj, 1);
            fs = d.fs;

            step = max(1, round(app.anim_speed * fs * 0.035));
            app.anim_idx = min(app.anim_idx + step, n);
            if app.anim_idx >= n, app.anim_idx = 1; end
            render_frame(app);
        end

        function render_frame(app)
            if ~isvalid(app) || isempty(app.data), return; end
            d = app.data;
            traj = d.trajectory;
            n = size(traj, 1);
            fs = d.fs;
            idx = app.anim_idx;try
                if isvalid(app.h_source)
                    app.h_source.XData = traj(idx,1);
                    app.h_source.YData = traj(idx,2);
                    app.h_source.ZData = traj(idx,3);
                end
                if isvalid(app.h_cursor3d)
                    app.h_cursor3d.XData = [traj(idx,1) traj(idx,1)];
                    app.h_cursor3d.YData = [traj(idx,2) traj(idx,2)];
                end
            catch, return; end

            % ???????
            if isfield(d, 'head_yaw_deg') && ~isempty(app.h_head_dir) && isvalid(app.h_head_dir)
                yaw_h = deg2rad(d.head_yaw_deg(idx));
                fwd = d.head_radius * 1.5;
                hc_h = d.head_center;
                app.h_head_dir.XData = [hc_h(1) hc_h(1)+fwd*sin(yaw_h)];
                app.h_head_dir.YData = [hc_h(2) hc_h(2)+fwd*cos(yaw_h)];
                app.h_head_dir.ZData = [hc_h(3) hc_h(3)];
            end

            update_overlay(app, traj(idx,:));

            tc = idx / fs;
            if isvalid(app.AxesWave)
                hold(app.AxesWave, 'on');
                yl = ylim(app.AxesWave);
                try delete(app.h_wave_cur); catch; end
                app.h_wave_cur = plot(app.AxesWave, [tc tc], yl, ...
                    'w-', 'LineWidth', 1.5);
                hold(app.AxesWave, 'off');
            end

            if isvalid(app.AxesAnal)
                hold(app.AxesAnal, 'on');
                yyaxis(app.AxesAnal, 'left');
                yl2 = ylim(app.AxesAnal);
                try delete(app.h_anal_cur); catch; end
                app.h_anal_cur = plot(app.AxesAnal, [tc tc], yl2, ...
                    'w-', 'LineWidth', 1.5);
                hold(app.AxesAnal, 'off');
            end

            if idx >= n, app.anim_idx = 1; end
            if isvalid(app.ProgressSlider)
                app.ProgressSlider.Value = app.anim_idx / max(n, 1);
            end
        end

        function on_reset(app, ~)
            if ~isvalid(app), return; end
            stop_anim(app);
            app.anim_idx = 1;
            if isvalid(app.ProgressSlider)
                app.ProgressSlider.Value = 0;
            end
            if ~isempty(app.data)
                render_3d(app);
                status(app, '动画已重置。', [0.3 0.3 0.3]);
            end
        end

        function on_wave_channel(app)
            if ~isvalid(app) || isempty(app.data), return; end
            render_wave(app);
            if isfield(app.data, 'stereo_raw')
                render_wave_raw(app);
            end
        end

        function on_anal_zoom(app)
            if ~isvalid(app) || isempty(app.data), return; end
            ax = app.AxesAnal;
            yyaxis(ax, 'left');
            ylim(ax, [-app.ItdZoom.Value app.ItdZoom.Value]);
            yyaxis(ax, 'right');
            ylim(ax, [-app.IldZoom.Value app.IldZoom.Value]);
        end

        function on_speed(app, ~)
            if ~isvalid(app), return; end
            app.anim_speed = app.SpeedSlider.Value;
            if app.anim_speed >= 1
                app.SpeedVal.Text = sprintf('%.0fx', app.anim_speed);
            else
                app.SpeedVal.Text = sprintf('%.1fx', app.anim_speed);
            end
        end

        function on_progress(app, ~)
            if ~isvalid(app) || isempty(app.data), return; end
            n = size(app.data.trajectory, 1);
            if n == 0, return; end
            app.anim_idx = max(1, round(app.ProgressSlider.Value * n));
            render_frame(app);
        end
    end
end

% ============ 本地函数: YAML 场景预览解析 ============
function preview = parse_yaml_preview(yaml_path)
    preview = [];
    if ~exist(yaml_path, 'file'), return; end

    try
        fid = fopen(yaml_path, 'r');
        if fid < 0, return; end
        raw = textscan(fid, '%s', 'Delimiter', '\n', 'Whitespace', '');
        fclose(fid);
        lines = raw{1};
    catch
        return;
    end

    % 默认值
    room_dims = [5 4 3];
    head_center = [2.5 2.0 1.5];
    head_radius = 0.09;
    left_ear = [];
    right_ear = [];
    mot_type = 'static';
    mot_center = [2.5 2.0 1.5];
    mot_radius = 1.5;
    mot_start = 0;
    mot_end = 180;
    mot_speed = 0.5;
    scene_name = '';

    in_motion = false;
    for i = 1:length(lines)
        line_orig = lines{i};
        line = strtrim(line_orig);
        if isempty(line) || startsWith(line, '#'), continue; end

        indent = length(line_orig) - length(line);
        if indent == 0, in_motion = startsWith(line, 'motion:'); end

        % room: dimensions: [...]
        if contains(line, 'dimensions:')
            room_dims = parse_array(line);
        end

        % head_center: [...]
        if contains(line, 'head_center:')
            head_center = parse_array(line);
        end

        % head_radius:
        if startsWith(line, 'head_radius:')
            head_radius = parse_scalar(line);
        end

        % left_ear: [...]
        if contains(line, 'left_ear:')
            left_ear = parse_array(line);
        end

        % right_ear: [...]
        if contains(line, 'right_ear:')
            right_ear = parse_array(line);
        end

        % name:
        if startsWith(line, 'name:')
            parts = strsplit(line, ':');
            if length(parts) >= 2
                scene_name = strtrim(parts{2});
            end
        end

        if in_motion
            if startsWith(line, 'type:')
                parts = strsplit(line, ':');
                if length(parts) >= 2, mot_type = strtrim(parts{2}); end
            end
            if contains(line, 'center:')
                mot_center = parse_array(line);
            end
            if startsWith(line, 'radius:')
                val = parse_scalar(line); if val > 0, mot_radius = val; end
            end
            if startsWith(line, 'start_angle:')
                mot_start = parse_scalar(line);
            end
            if startsWith(line, 'end_angle:')
                mot_end = parse_scalar(line);
            end
            if startsWith(line, 'speed:')
                mot_speed = parse_scalar(line);
            end
        end
    end

    % 计算耳朵位置（如果 YAML 中未指定）
    if isempty(left_ear)
        ear_spacing = head_radius * 0.83;
        left_ear = [head_center(1)-ear_spacing, head_center(2), head_center(3)];
        right_ear = [head_center(1)+ear_spacing, head_center(2), head_center(3)];
    end

    % 生成轨迹预览点
    traj_preview = [];
    n_pts = 180;
    switch mot_type
        case 'static'
            traj_preview = repmat(mot_center, [1 1]);
        case 'linear'
            traj_preview = [mot_center; mot_center + [mot_radius 0 0]];
        case 'semicircle'
            angles = linspace(deg2rad(mot_start), deg2rad(mot_end), n_pts);
            traj_preview = [mot_center(1)+mot_radius*cos(angles)', ...
                           mot_center(2)+mot_radius*sin(angles)', ...
                           repmat(mot_center(3), n_pts, 1)];
        case 'circle'
            angles = linspace(0, 2*pi, n_pts);
            traj_preview = [mot_center(1)+mot_radius*cos(angles)', ...
                           mot_center(2)+mot_radius*sin(angles)', ...
                           repmat(mot_center(3), n_pts, 1)];
    end

    preview = struct(...
        'name', scene_name, ...
        'room_dims', room_dims, ...
        'head_center', head_center, ...
        'head_radius', head_radius, ...
        'left_ear', left_ear, ...
        'right_ear', right_ear, ...
        'traj_preview', traj_preview);
end

function arr = parse_array(line)
    arr = [0 0 0];
    try
        st = strfind(line, '[');
        en = strfind(line, ']');
        if isempty(st) || isempty(en), return; end
        inner = line(st(1)+1:en(1)-1);
        parts = strsplit(strtrim(inner), ',');
        for k = 1:min(3, length(parts))
            arr(k) = str2double(strtrim(parts{k}));
        end
    catch
    end
end

function val = parse_scalar(line)
    val = 0;
    try
        parts = strsplit(line, ':');
        if length(parts) >= 2
            val = str2double(strtrim(parts{2}));
            if isnan(val), val = 0; end
        end
    catch
    end
end

function [vtx, fcs] = load_head_model()
    persistent cached_v cached_f;
    if ~isempty(cached_v), vtx = cached_v; fcs = cached_f; return; end
    vtx = []; fcs = [];
    paths = {'D:\shengxuedingwei2\matlab\MaleHead.obj', ...
             'D:\shengxuedingwei\config\microphone\MaleHead.obj'};
    for c = 1:length(paths)
        p = paths{c}; if ~exist(p, 'file'), continue; end
        fid = fopen(p, 'r'); if fid == -1, continue; end
        vtxList = []; faceList = {};
        while ~feof(fid)
            ln = fgetl(fid);
            if ~ischar(ln), break; end
            if isempty(ln) || ln(1) == '#', continue; end
            parts = strsplit(strtrim(ln));
            if isempty(parts), continue; end
            if strcmp(parts{1}, 'v') && length(parts) >= 4
                vtxList(end+1,:) = [str2double(parts{2}) str2double(parts{3}) str2double(parts{4})];
            elseif strcmp(parts{1}, 'f')
                fv = [];
                for i = 2:length(parts)
                    ip = strsplit(parts{i}, '/');
                    idx = str2double(ip{1});
                    if idx < 0, idx = size(vtxList,1) + idx + 1; end
                    fv(end+1) = idx;
                end
                if ~isempty(fv), faceList{end+1} = fv; end
            end
        end
        fclose(fid);
        if isempty(vtxList), continue; end
        for i = 1:length(faceList)
            fv = faceList{i};
            if length(fv) == 3, fcs(end+1,:) = fv;
            elseif length(fv) > 3
                for j = 2:length(fv)-1, fcs(end+1,:) = [fv(1) fv(j) fv(j+1)]; end
            end
        end
        if ~isempty(fcs)
            vtxList = vtxList - mean(vtxList);
            ext = max(vtxList) - min(vtxList);
            scl = max(ext)/2; if scl < 0.01, scl = 0.09; end
            vtx = vtxList / scl * 0.09;
            vtx = vtx(:, [1 3 2]);  % ZBrush(Y-up) → MATLAB(Z-up)
            cached_v = vtx; cached_f = fcs; return;
        end
    end
end

